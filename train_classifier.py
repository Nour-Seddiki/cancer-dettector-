"""Week 2 / stage (a): fine-tune DenseNet121 for 14-way multi-label classification.

A self-contained, evaluable milestone *before* the decoder exists: if per-class AUROC here
is at chance, the problem is the image pipeline, not the language model. Reports per-class
AUROC (CheXNet's own metric) plus a mean over the classes with enough positives to be
meaningful.

Runs entirely on the local GPU (plan Section 5).

    python train_classifier.py --epochs 15 --batch-size 32
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cnn_model import DenseNet121Encoder
from dataloader import make_classification_loaders, read_manifest, split_rows, label_vector
from labels import CONDITIONS
from utils import (AverageMeter, EarlyStopping, cap_gpu_memory, cosine_lr, format_seconds,
                   get_device, keep_awake, save_checkpoint)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
MIN_POS_FOR_AUROC = 10  # below this the AUROC estimate is too noisy to average in


def pos_weight_from_rows(rows):
    """BCE positive weighting - most CXRs are normal, so rare findings need up-weighting.

    Clamped at 10x: without a cap, a condition with 5 positives in 3000 studies gets a
    600x weight and its gradient drowns out everything else.
    """
    labels = np.array([label_vector(r) for r in rows], dtype=np.float32)
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(weight, 1.0, 10.0), dtype=torch.float32), pos


def auroc(y_true, y_score):
    """Per-class AUROC via the rank (Mann-Whitney U) identity - no sklearn dependency.

    Returns NaN for a class with no positives or no negatives, where AUROC is undefined.
    """
    out = []
    for c in range(y_true.shape[1]):
        t, s = y_true[:, c], y_score[:, c]
        n_pos, n_neg = int(t.sum()), int((1 - t).sum())
        if n_pos == 0 or n_neg == 0:
            out.append(float("nan"))
            continue
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s), dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        # average ranks within ties, otherwise ties bias the statistic
        sorted_s = s[order]
        i = 0
        while i < len(sorted_s):
            j = i
            while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
            i = j + 1
        out.append((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return np.array(out)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_true, all_score = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits.float(), targets)
        loss_meter.update(loss.item(), images.size(0))
        all_true.append(targets.cpu().numpy())
        all_score.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()

    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    per_class = auroc(y_true, y_score)
    support = y_true.sum(axis=0)
    usable = (support >= MIN_POS_FOR_AUROC) & ~np.isnan(per_class)
    mean_auroc = float(per_class[usable].mean()) if usable.any() else float("nan")
    return loss_meter.avg, per_class, support, mean_auroc, usable


def print_auroc_table(per_class, support, usable):
    print(f"    {'condition':28s} {'AUROC':>7s} {'n_pos':>7s}")
    for i, c in enumerate(CONDITIONS):
        mark = " " if usable[i] else "*"
        val = "  n/a  " if np.isnan(per_class[i]) else f"{per_class[i]:7.4f}"
        print(f"    {c:28s} {val} {int(support[i]):7d} {mark}")
    print(f"    (* = fewer than {MIN_POS_FOR_AUROC} positives, excluded from the mean)")


def main():
    p = argparse.ArgumentParser(description="Stage (a): DenseNet121 multi-label baseline.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clahe", action="store_true", help="CLAHE contrast enhancement")
    p.add_argument("--freeze-epochs", type=int, default=1,
                   help="epochs with the backbone frozen before unfreezing (head warmup)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gpu-fraction", type=float, default=0.8)
    p.add_argument("--out", default=str(CKPT / "classifier.pt"))
    args = p.parse_args()

    device = get_device()
    print(f"device: {device}")
    cap_gpu_memory(args.gpu_fraction, device)
    keep_awake(True)

    rows = read_manifest()
    train_rows = split_rows(rows, "train")
    print(f"train images: {len(train_rows)}, val images: {len(split_rows(rows, 'val'))}")

    train_loader, val_loader = make_classification_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, clahe=args.clahe, rows=rows)

    model = DenseNet121Encoder(pretrained=True).to(device)
    pos_weight, pos_counts = pos_weight_from_rows(train_rows)
    print("positive counts (train):",
          {c: int(n) for c, n in zip(CONDITIONS, pos_counts) if n})
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    max_iters = args.epochs * steps_per_epoch
    warmup_iters = int(args.warmup_frac * max_iters)
    stopper = EarlyStopping(patience=args.patience, mode="max", min_delta=1e-4)

    # Head warmup: a randomly-initialised head sends garbage gradients into pretrained
    # features for the first few hundred steps. Freeze, let the head settle, then unfreeze.
    if args.freeze_epochs > 0:
        model.freeze_backbone()
        print(f"backbone frozen for the first {args.freeze_epochs} epoch(s)")

    print(f"\ntraining for {args.epochs} epochs ({max_iters} steps)")
    start = time.time()
    it = 0
    best_auroc = float("-inf")

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            model.unfreeze_all()
            print("backbone unfrozen")

        epoch_loss = AverageMeter()
        for images, targets in train_loader:
            lr = cosine_lr(it, args.lr, warmup_iters, max_iters)
            for g in optimizer.param_groups:
                g["lr"] = lr

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits.float(), targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.update(loss.item(), images.size(0))
            it += 1

        val_loss, per_class, support, mean_auroc, usable = evaluate(
            model, val_loader, criterion, device)
        elapsed = format_seconds(time.time() - start)
        print(f"\nepoch {epoch + 1}/{args.epochs}  train loss {epoch_loss.avg:.4f}  "
              f"val loss {val_loss:.4f}  mean AUROC {mean_auroc:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  [{elapsed}]")
        print_auroc_table(per_class, support, usable)

        improved, should_stop = stopper.step(mean_auroc)
        if improved:
            best_auroc = mean_auroc
            save_checkpoint({
                "model": model.state_dict(),
                "epoch": epoch,
                "mean_auroc": mean_auroc,
                "per_class_auroc": per_class.tolist(),
                "conditions": CONDITIONS,
                "args": vars(args),
            }, args.out)
            print(f"    saved (best mean AUROC {mean_auroc:.4f}) -> {args.out}")
        if should_stop:
            print(f"\nearly stop: no improvement for {args.patience} epochs")
            break

    keep_awake(False)
    print(f"\ndone in {format_seconds(time.time() - start)}. "
          f"best mean AUROC {best_auroc:.4f}, checkpoint {args.out}")
    print("Next: python training.py --cnn-checkpoint " + args.out)


if __name__ == "__main__":
    main()
