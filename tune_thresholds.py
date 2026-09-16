"""Tune per-condition thresholds for a label-token model's classifier head, on val.

    python tune_thresholds.py --checkpoint checkpoints/rg_v5_phase2.pt

Runs only the CNN + classifier head over the split and picks the per-condition thresholds
that maximise the head's micro-averaged F1 over the 13 abnormal conditions, scored against
the conditions the rule-based labeller reads off the reference reports - the same labeller
and the same average as the clinical-efficacy metric. They are stored in the checkpoint
under "label_thresholds"; from then on evaluate.py, generate.py and demo.py feed the
decoder binary condition tokens instead of soft probabilities.

Jointly on micro F1, not one F1 per condition: a rare condition the head barely separates
has its best per-condition F1 at a threshold that flags a third of all studies at ~4%
precision (Pneumonia on val). Each of those studies then gets the finding written into
its report, which costs far more micro precision than the condition's few hits add back.
The joint search switches such conditions off instead, while keeping common ones like
Lung Opacity whose precision is modest but well above what they cost.

Since v12 the default is one threshold shared by all 13 abnormal conditions (`--rule
global`). Thirteen free parameters fitted on 361 val studies do not survive the move to
another split - they cost about 0.06 head micro F1 - and sharing one threshold is worth
+0.010 to +0.014 test CE micro F1 at completely unchanged model weights. `--rule coord`
keeps the old per-condition behaviour.

Tune on val, report on test - never tune on the split you report. Note the corollary: a val
CE score is partly *in-sample* for whatever thresholds this script fitted, so two threshold
rules cannot be compared on it. Fit on one random half of val and score on the other.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import CXRReportDataset, make_collate_fn, read_manifest, split_rows
from evaluate import load_generator
from labels import CONDITIONS, NO_FINDING_IDX
from utils import get_device, save_checkpoint

GRID = np.arange(0.05, 0.96, 0.025)
NEVER = 1.01  # above any sigmoid output: the condition token is always "absent"
ABNORMAL = [c for c in range(len(CONDITIONS)) if c != NO_FINDING_IDX]


def counts(p, y, t):
    """(tp, fp, fn) of thresholding scores `p` at `t` against 0/1 labels `y`."""
    pred = p >= t
    return np.array([np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))])


def f1_score(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def best_single(p, y):
    """Per-condition F1-optimal threshold (NEVER if nothing scores above 0)."""
    best_f1, best_t = 0.0, NEVER
    for t in GRID:
        f1 = f1_score(*counts(p, y, t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def tune_micro(probs, y, passes=5):
    """Coordinate ascent on micro F1 over the abnormal conditions.

    Starts from each condition's own F1-optimal threshold, then repeatedly re-picks one
    condition's threshold (NEVER included) with all the others held fixed, until a full
    pass changes nothing.
    """
    thr = {c: best_single(probs[:, c], y[:, c]) for c in ABNORMAL}
    cnt = {c: counts(probs[:, c], y[:, c], thr[c]) for c in ABNORMAL}
    for _ in range(passes):
        changed = False
        for c in ABNORMAL:
            rest = sum(cnt[k] for k in ABNORMAL if k != c)
            best_f1, best_t, best_cnt = f1_score(*(rest + cnt[c])), thr[c], cnt[c]
            for t in [*GRID, NEVER]:
                cc = counts(probs[:, c], y[:, c], t)
                f1 = f1_score(*(rest + cc))
                if f1 > best_f1 + 1e-9:
                    best_f1, best_t, best_cnt = f1, float(t), cc
            changed |= best_t != thr[c]
            thr[c], cnt[c] = best_t, best_cnt
        if not changed:
            break
    return thr


def tune_global(probs, y):
    """One threshold shared by all 13 abnormal conditions.

    Thirteen thresholds coordinate-ascended on 361 val studies do not survive the move to
    another split: they cost about 0.06 head micro F1 going from val to test, and several
    conditions' optima move a long way (pleural effusion 0.575 to 0.350, atelectasis 0.700
    to 0.475). Sharing one threshold is the crudest way to cut that variance, and it is the
    rule that won a fit-on-half-of-val, score-on-the-other-half comparison: +0.020 held-out
    micro F1 over `tune_micro` on v5a and +0.013 on v11, against bagging (+0.006 / +0.003)
    and forcing low-support conditions off (negative for both). See IMPROVEMENT_LOG.md.
    """
    best_f1, best_t = -1.0, NEVER
    for t in GRID:
        total = sum(counts(probs[:, c], y[:, c], t) for c in ABNORMAL)
        f1 = f1_score(*total)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return {c: best_t for c in ABNORMAL}


def tune_shrunk(probs, y, alpha=0.5):
    """Per-condition thresholds pulled `1 - alpha` of the way toward the global one.

    The middle ground between the two: it keeps some per-condition freedom while paying
    less of its variance. On held-out val halves it landed between them, close to `global`.
    """
    coord, glob = tune_micro(probs, y), tune_global(probs, y)
    return {c: alpha * coord[c] + (1 - alpha) * glob[c] for c in ABNORMAL}


@torch.no_grad()
def head_probs(model, tokenizer, rows, device, batch_size=32, num_workers=2):
    """Head probabilities and reference-text labels for every study in `rows`."""
    ds = CXRReportDataset(rows, tokenizer, train=False, max_len=model.block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=make_collate_fn(tokenizer.pad_id))
    probs, text_labels = [], []
    for images, _, _, labels in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            p = model.head_probabilities(images.to(device))
        probs.append(p.float().cpu().numpy())
        text_labels.append(labels[:, 1].numpy())
    return np.concatenate(probs), np.concatenate(text_labels)


def main():
    p = argparse.ArgumentParser(description="Tune binary condition-token thresholds on val.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--no-finding", choices=["derived", "threshold"], default="derived",
                   help="derive the No Finding token from the abnormal ones (in every "
                        "ground-truth label the decoder saw it means 'none of them'), or "
                        "threshold it on its own")
    p.add_argument("--rule", choices=["global", "coord", "shrunk"], default="global",
                   help="how to fit the thresholds: one threshold shared by all abnormal "
                        "conditions (the default), per-condition coordinate ascent on micro "
                        "F1 (the original), or the two blended. `global` is the default "
                        "because the 13-parameter fit overfits val badly enough to cost "
                        "~0.06 head micro F1 on test, while sharing one threshold is worth "
                        "+0.010 to +0.014 test CE micro F1 at unchanged model weights.")
    p.add_argument("--shrink-alpha", type=float, default=0.5,
                   help="with --rule shrunk: weight on the per-condition thresholds")
    p.add_argument("--tta-zoom", type=float, default=0.0,
                   help="average the head's probabilities over the original view plus 5 "
                        "crops (centre + corners) of a copy upscaled by this factor "
                        "(0 = off). Stored in the checkpoint, so evaluate.py and "
                        "generate.py reproduce the probabilities these thresholds were "
                        "tuned on.")
    args = p.parse_args()

    device = get_device()
    model, tokenizer = load_generator(args.checkpoint, device)
    if model.label_bridge is None:
        raise SystemExit("this checkpoint has no condition tokens (trained without --label-tokens)")
    model.label_thresholds = None  # tune on the raw probabilities
    model.tta_zoom = args.tta_zoom
    if args.tta_zoom:
        print(f"  head TTA: original view + 5 crops at zoom {args.tta_zoom:g}")

    rows = split_rows(read_manifest(), args.split)
    probs, y = head_probs(model, tokenizer, rows, device)

    if args.rule == "global":
        thr = tune_global(probs, y)
    elif args.rule == "shrunk":
        thr = tune_shrunk(probs, y, args.shrink_alpha)
    else:
        thr = tune_micro(probs, y)
    thresholds = [thr.get(c, 0.0) for c in range(len(CONDITIONS))]
    # NaN tells model.encode to derive No Finding from the abnormal tokens.
    thresholds[NO_FINDING_IDX] = (float("nan") if args.no_finding == "derived"
                                  else best_single(probs[:, NO_FINDING_IDX], y[:, NO_FINDING_IDX]))

    print(f"\n  {'condition':28s} {'thresh':>7s} {'P':>6s} {'R':>6s} {'F1':>6s} "
          f"{'flagged':>8s} {'support':>8s}")
    total = np.zeros(3)
    for c, name in enumerate(CONDITIONS):
        if c == NO_FINDING_IDX and args.no_finding == "derived":
            print(f"  {name:28s} {'derived':>7s}")
            continue
        tp, fp, fn = cnt = counts(probs[:, c], y[:, c], thresholds[c])
        if c != NO_FINDING_IDX:
            total += cnt
        shown = "never" if thresholds[c] == NEVER else f"{thresholds[c]:.3f}"
        print(f"  {name:28s} {shown:>7s} {tp / max(1, tp + fp):6.3f} {tp / max(1, tp + fn):6.3f} "
              f"{f1_score(tp, fp, fn):6.3f} {100 * (tp + fp) / len(y):7.1f}% {int(y[:, c].sum()):8d}")
    tp, fp, fn = total
    print(f"  {'-' * 72}\n  {'head micro (abnormal)':28s} {'':7s} {tp / max(1, tp + fp):6.3f} "
          f"{tp / max(1, tp + fn):6.3f} {f1_score(tp, fp, fn):6.3f}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt["label_thresholds"] = thresholds
    ckpt["label_thresholds_split"] = args.split
    ckpt["label_threshold_rule"] = args.rule
    ckpt["tta_zoom"] = args.tta_zoom
    save_checkpoint(ckpt, args.checkpoint)
    print(f"\nstored thresholds (rule {args.rule}, tuned on {args.split}, "
          f"No Finding {args.no_finding}"
          + (f", head TTA zoom {args.tta_zoom:g}" if args.tta_zoom else "")
          + f") in {args.checkpoint}")


if __name__ == "__main__":
    main()
