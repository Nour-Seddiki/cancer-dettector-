"""Worked example: run the trained models on held-out chest X-rays and draw the result.

    python demo.py
    python demo.py --checkpoint checkpoints/rg_w3_phase2.pt --classifier checkpoints/classifier.pt
    python demo.py --conditions "No Finding" Cardiomegaly Edema --out outputs/demo_edema.png

For each condition it takes the first test-split study carrying it, then shows side by side:
  * the X-ray the model actually sees (224x224, eval transform)
  * Grad-CAM of the stage (a) classifier for that condition - where it is looking
  * radiologist report vs. generated report, the conditions the rule-based labeller finds
    in the generated one, and the classifier's top-4 scores
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from cnn_model import load_encoder
from dataloader import DATA, IMAGENET_MEAN, IMAGENET_STD, label_vector, read_manifest, split_rows
from evaluate import load_generator
from generate import generate_report, prepare_image
from labels import CONDITIONS, labels_from_text, positives
from utils import get_device

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
DEFAULT_CONDITIONS = ["No Finding", "Cardiomegaly", "Pleural Effusion", "Pneumothorax"]


def pick_examples(rows, conditions):
    """First study in `rows` carrying each condition, no study used twice (deterministic)."""
    picked, used = [], set()
    for cond in conditions:
        ci = CONDITIONS.index(cond)
        for r in rows:
            if r["uid"] not in used and label_vector(r)[ci]:
                picked.append((cond, r))
                used.add(r["uid"])
                break
        else:
            print(f"  no study with {cond!r} in this split, skipping")
    return picked


def grad_cam(encoder, x, cls_idx):
    """Grad-CAM on the (1024, 7, 7) DenseNet map for one class logit -> (224, 224) in [0, 1]."""
    encoder.zero_grad(set_to_none=True)
    with torch.enable_grad():
        fmap = encoder.forward_map(x)
        fmap.retain_grad()
        logit = encoder.classifier(F.adaptive_avg_pool2d(fmap, 1).flatten(1))[0, cls_idx]
        logit.backward()
    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()


def to_display(x):
    """Normalised (1, 3, 224, 224) tensor -> (224, 224) grayscale array for imshow."""
    img = x[0, 0].cpu() * IMAGENET_STD[0] + IMAGENET_MEAN[0]
    return img.clamp(0, 1).numpy()


def wrap(text, width=78, limit=330):
    text = text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."
    return "\n".join(textwrap.wrap(text, width))


def main():
    p = argparse.ArgumentParser(description="Report generation + classifier demo on test X-rays.")
    p.add_argument("--checkpoint", default=str(CKPT / "rg_v12b_global_tta.pt"))
    p.add_argument("--classifier", default=str(CKPT / "classifier_v2.pt"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=CONDITIONS,
                   metavar="CONDITION")
    p.add_argument("--beam-size", type=int, default=1, help="1 = greedy")
    p.add_argument("--out", default=str(ROOT / "outputs" / "xray_demo.png"))
    args = p.parse_args()

    device = get_device()
    print(f"device: {device}")

    generator, tokenizer = load_generator(args.checkpoint, device)
    cls_ckpt = torch.load(args.classifier, map_location="cpu", weights_only=False)
    classifier = load_encoder(args.classifier, pretrained=False, device=device, strict=True).eval()
    print(f"loaded {args.classifier} (epoch {cls_ckpt['epoch']}, "
          f"val mean AUROC {cls_ckpt['mean_auroc']:.4f})")

    examples = pick_examples(split_rows(read_manifest(), args.split), args.conditions)
    if not examples:
        raise SystemExit("nothing to show")

    fig, axes = plt.subplots(len(examples), 3, figsize=(17, 4.6 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 1, 2.3]}, squeeze=False)
    for row_i, (cond, row) in enumerate(examples):
        x = prepare_image(DATA / row["image_path"]).to(device)
        truth = positives(label_vector(row))

        report = generate_report(generator, tokenizer, x, device, beam_size=args.beam_size)[0]
        gen_labels = positives(labels_from_text(report))

        with torch.no_grad():
            probs = torch.sigmoid(classifier(x).float())[0].cpu()
        ranked = probs.argsort(descending=True).tolist()

        cam_idx = CONDITIONS.index(cond) if cond != "No Finding" else ranked[0]
        cam = grad_cam(classifier, x, cam_idx)

        print(f"\n=== {row['uid']} ({row['image_id']})  picked for: {cond} ===")
        print(f"  ground truth     : {truth}")
        print(f"  radiologist      : {row['report']}")
        print(f"  generated report : {report}")
        print(f"  labels in report : {gen_labels or ['(none)']}")
        print("  classifier scores:")
        for i in ranked:
            mark = "  <- truth" if CONDITIONS[i] in truth else ""
            print(f"    {CONDITIONS[i]:28s} {probs[i]:.3f}{mark}")

        img = to_display(x)
        ax_img, ax_cam, ax_txt = axes[row_i]
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"{row['uid']}  ({args.split} split)", fontsize=11)
        ax_cam.imshow(img, cmap="gray")
        ax_cam.imshow(cam, cmap="jet", alpha=0.4)
        ax_cam.set_title(f"Grad-CAM: {CONDITIONS[cam_idx]}", fontsize=11)
        for a in (ax_img, ax_cam):
            a.axis("off")

        cls_line = "   ".join(f"{CONDITIONS[i]} {probs[i]:.2f}" for i in ranked[:4])
        ax_txt.axis("off")
        ax_txt.text(0, 1, f"Ground truth: {', '.join(truth)}", fontsize=11.5,
                    fontweight="bold", va="top", transform=ax_txt.transAxes)
        ax_txt.text(0, 0.88, "Radiologist report:\n" + wrap(row["report"]), fontsize=9.5,
                    va="top", family="monospace", transform=ax_txt.transAxes)
        ax_txt.text(0, 0.50, "Model report:\n" + wrap(report), fontsize=9.5, va="top",
                    family="monospace", color="#1a4f9c", transform=ax_txt.transAxes)
        ax_txt.text(0, 0.17, f"Conditions in model report: {', '.join(gen_labels) or '(none)'}",
                    fontsize=10, va="top", transform=ax_txt.transAxes)
        ax_txt.text(0, 0.07, f"Classifier top-4: {cls_line}", fontsize=10, va="top",
                    transform=ax_txt.transAxes)

    fig.suptitle(f"Chest X-ray report generation - {Path(args.checkpoint).name} + "
                 f"{Path(args.classifier).name}", fontsize=14, y=0.995)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
