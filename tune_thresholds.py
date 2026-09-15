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

Tune on val, report on test - never tune on the split you report.
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
            _, logits = model.encode(images.to(device))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
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
    args = p.parse_args()

    device = get_device()
    model, tokenizer = load_generator(args.checkpoint, device)
    if model.label_bridge is None:
        raise SystemExit("this checkpoint has no condition tokens (trained without --label-tokens)")
    model.label_thresholds = None  # tune on the raw probabilities

    rows = split_rows(read_manifest(), args.split)
    probs, y = head_probs(model, tokenizer, rows, device)

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
    save_checkpoint(ckpt, args.checkpoint)
    print(f"\nstored thresholds (tuned on {args.split}, No Finding {args.no_finding}) "
          f"in {args.checkpoint}")


if __name__ == "__main__":
    main()
