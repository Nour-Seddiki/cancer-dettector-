"""Week 6: generate reports for a split and score them.

    python evaluate.py --checkpoint checkpoints/report_generator.pt --split test --beam-size 3

Prints BLEU-1..4 / ROUGE-L / CIDEr (fast iteration signal), the clinical-efficacy
precision/recall/F1 (the metric that actually matters), degeneration diagnostics, and the
worst-scoring examples for error analysis. Dumps every generation to JSON so the error
analysis can be done outside this script.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import (CXRReportDataset, ReportTokenizer, make_collate_fn, read_manifest,
                        split_rows)
from labels import CONDITIONS
from metrics import clinical_efficacy, diversity_stats, nlg_metrics, rouge_l
from model import ReportGenerator
from utils import cap_gpu_memory, format_seconds, get_device

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"


def load_generator(checkpoint_path, device):
    """Rebuild the model from the config stored in its checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = ReportTokenizer(ckpt["vocab"])

    model = ReportGenerator(
        vocab_size=cfg["vocab_size"], pad_id=cfg["pad_id"], n_emb=cfg["n_emb"],
        n_head=cfg["n_head"], n_layer=cfg["n_layer"], block_size=cfg["block_size"],
        dropout=cfg["dropout"], pretrained_cnn=False,
        factorized_pos=cfg.get("factorized_pos", False),
        label_tokens=cfg.get("label_tokens", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("label_thresholds") is not None:
        model.label_thresholds = torch.tensor(ckpt["label_thresholds"], device=device)
    # Head TTA is a property of the stored thresholds rather than a separate eval-time
    # choice: they were tuned on the averaged probabilities and do not transfer to the
    # single-view ones.
    model.tta_zoom = float(ckpt.get("tta_zoom") or 0.0)
    model.eval()
    print(f"loaded {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.4f}, vocab {len(tokenizer)}"
          + (", binary condition tokens" if model.label_thresholds is not None else "")
          + (f", head TTA zoom {model.tta_zoom:g}" if model.tta_zoom else "") + ")")
    return model, tokenizer


@torch.no_grad()
def generate_split(model, tokenizer, rows, device, batch_size=16, beam_size=3,
                   max_new_tokens=120, length_penalty=0.6, num_workers=2):
    ds = CXRReportDataset(rows, tokenizer, train=False, max_len=model.block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=make_collate_fn(tokenizer.pad_id), pin_memory=True)

    hypotheses = []
    start = time.time()
    for i, (images, *_) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model.generate(images, max_new_tokens, tokenizer.bos_id, tokenizer.eos_id,
                                 beam_size=beam_size, length_penalty=length_penalty)
        hypotheses.extend(tokenizer.decode(seq.tolist()) for seq in out)
        done = min((i + 1) * batch_size, len(ds))
        print(f"\r  generated {done}/{len(ds)}  [{format_seconds(time.time() - start)}]",
              end="", flush=True)
    print()
    return hypotheses


def print_ce_table(ce):
    print(f"\n  {'condition':28s} {'P':>7s} {'R':>7s} {'F1':>7s} {'support':>8s}")
    for name in CONDITIONS:
        s = ce["per_condition"][name]
        if s["support"] == 0 and s["fp"] == 0:
            continue
        print(f"  {name:28s} {s['precision']:7.3f} {s['recall']:7.3f} {s['f1']:7.3f} "
              f"{s['support']:8d}")
    print(f"  {'-' * 60}")
    print(f"  {'MICRO (excl. No Finding)':28s} {ce['micro_precision']:7.3f} "
          f"{ce['micro_recall']:7.3f} {ce['micro_f1']:7.3f}")
    print(f"  {'MACRO F1':28s} {'':7s} {'':7s} {ce['macro_f1']:7.3f}")


def worst_examples(hypotheses, references, rows, k=5):
    """Lowest ROUGE-L pairs - the starting point for the plan's week-6 error analysis."""
    scored = []
    for i, (h, r) in enumerate(zip(hypotheses, references)):
        s = rouge_l([h], [r])["ROUGE-L"]
        scored.append((s, i))
    scored.sort()
    return [(s, rows[i]["uid"], references[i], hypotheses[i]) for s, i in scored[:k]]


def main():
    p = argparse.ArgumentParser(description="Evaluate the report generator.")
    p.add_argument("--checkpoint", default=str(CKPT / "rg_v12b_global_tta.pt"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    # Greedy by default: beam search drifts to the highest-likelihood report, which here is
    # the normal template. On val it cut clinical-efficacy micro F1 from 0.20 to 0.06 (v2).
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--length-penalty", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N studies")
    p.add_argument("--no-coco", action="store_true",
                   help="always use the built-in metric implementations")
    p.add_argument("--out", default=str(ROOT / "outputs" / "eval.json"))
    args = p.parse_args()

    device = get_device()
    print(f"device: {device}")
    cap_gpu_memory(0.8, device)

    model, tokenizer = load_generator(args.checkpoint, device)

    rows = split_rows(read_manifest(), args.split)
    if args.limit:
        rows = rows[:args.limit]
    references = [r["report"] for r in rows]
    print(f"{args.split} split: {len(rows)} images, "
          f"{len({r['uid'] for r in rows})} studies\n")

    hypotheses = generate_split(
        model, tokenizer, rows, device, batch_size=args.batch_size,
        beam_size=args.beam_size, max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty, num_workers=args.num_workers)

    print("\n=== NLG metrics ===")
    nlg = nlg_metrics(hypotheses, references, prefer_coco=not args.no_coco)
    for k, v in nlg.items():
        print(f"  {k:10s} {v:.4f}")

    print("\n=== Clinical efficacy (rule-based labeller) ===")
    ce = clinical_efficacy(hypotheses, references)
    print_ce_table(ce)
    print("\n  NB: this is the dependency-free rule-based labeller, not CheXbert. Treat it")
    print("  as a directional signal; see the README for swapping in CheXbert.")

    print("\n=== Diversity / degeneration ===")
    div = diversity_stats(hypotheses)
    for k, v in div.items():
        print(f"  {k:22s} {v:.4f}")
    ref_div = diversity_stats(references)
    print(f"  (references: unique {ref_div['unique_report_frac']:.3f}, "
          f"mean length {ref_div['mean_length']:.1f})")

    print("\n=== Worst 5 by ROUGE-L ===")
    for score, uid, ref, hyp in worst_examples(hypotheses, references, rows):
        print(f"\n  [{uid}] ROUGE-L {score:.3f}")
        print(f"    ref: {ref[:220]}")
        print(f"    gen: {hyp[:220]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "split": args.split,
            "beam_size": args.beam_size,
            "nlg": nlg,
            "clinical_efficacy": ce,
            "diversity": div,
            "generations": [
                {"uid": r["uid"], "image_id": r["image_id"], "reference": ref, "generated": hyp}
                for r, ref, hyp in zip(rows, references, hypotheses)
            ],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
