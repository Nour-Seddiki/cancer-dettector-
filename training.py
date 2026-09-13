"""Weeks 4-5 / stage (c): train the CNN -> Transformer report generator.

Two phases, in the order the plan prescribes (Section 3):
  1. CNN frozen. The decoder learns the language of radiology reports against stable
     visual features. Cheap, and it cannot overfit the backbone to ~3k studies.
  2. `denseblock4` + `norm5` unfrozen at a much lower LR (discriminative LRs, CNN <<
     decoder) once the decoder is emitting coherent text.

Debug first, then scale (plan weeks 4-5):

    # does the architecture learn at all? should reach near-zero loss in a few hundred steps
    python training.py --overfit 64 --epochs 200 --batch-size 8 --num-workers 0

    # phase 1, full IU X-Ray, local
    python training.py --epochs 30 --cnn-checkpoint checkpoints/classifier.pt

    # phase 2, joint fine-tune from the phase-1 checkpoint
    python training.py --epochs 15 --resume checkpoints/report_generator.pt \
        --unfreeze-cnn --lr 1e-4 --cnn-lr 1e-5
"""

import argparse
import time
from pathlib import Path

import torch

from cnn_model import load_encoder
from dataloader import make_report_loaders
from model import ReportGenerator, block_size, count_parameters, dropout, n_emb, n_head, n_layer
from utils import (AverageMeter, EarlyStopping, cap_gpu_memory, cosine_lr, format_seconds,
                   get_device, keep_awake, save_checkpoint)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"


@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=None):
    model.eval()
    meter = AverageMeter()
    for i, (images, tgt_in, tgt_out) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        tgt_in = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            _, loss = model(images, tgt_in, tgt_out)
        meter.update(loss.item(), images.size(0))
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meter.avg


@torch.no_grad()
def sample_reports(model, loader, tokenizer, device, n=2, max_new_tokens=100):
    """Print a couple of generated/reference pairs - the fastest read on whether the model
    is producing radiology-shaped text or degenerate repetition."""
    model.eval()
    images, tgt_in, tgt_out = next(iter(loader))
    images = images[:n].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type == "cuda")):
        generated = model.generate(images, max_new_tokens, tokenizer.bos_id, tokenizer.eos_id)
    model.train()

    for i in range(min(n, images.size(0))):
        ref_ids = [t for t in tgt_out[i].tolist() if t >= 0]
        print(f"    [ref] {tokenizer.decode(ref_ids)[:200]}")
        print(f"    [gen] {tokenizer.decode(generated[i].tolist())[:200]}")
        print()


def build_model(args, tokenizer, device):
    encoder = None
    if args.cnn_checkpoint:
        print(f"  warm-starting the CNN from {args.cnn_checkpoint}")
        encoder = load_encoder(args.cnn_checkpoint, pretrained=True)

    model = ReportGenerator(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        n_emb=args.n_emb, n_head=args.n_head, n_layer=args.n_layer,
        block_size=args.block_size, dropout=args.dropout,
        encoder=encoder, pretrained_cnn=True,
        factorized_pos=args.factorized_pos,
    ).to(device)

    model.freeze_cnn()
    if args.unfreeze_cnn:
        model.unfreeze_cnn_last_block()
        print("  CNN: denseblock4 + norm5 unfrozen (joint fine-tune)")
    else:
        print("  CNN: fully frozen (phase 1)")
    return model


def main():
    p = argparse.ArgumentParser(description="Stage (c): train the report generator.")
    # data
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=block_size)
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--overfit", type=int, default=0,
                   help="train on only N examples (architecture debugging, plan week 4)")
    # model
    p.add_argument("--n-emb", type=int, default=n_emb)
    p.add_argument("--n-head", type=int, default=n_head)
    p.add_argument("--n-layer", type=int, default=n_layer)
    p.add_argument("--block-size", type=int, default=block_size)
    p.add_argument("--dropout", type=float, default=dropout)
    p.add_argument("--factorized-pos", action="store_true",
                   help="row/col-factored grid positional embedding instead of flat")
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cnn-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=6)
    # phases / io
    p.add_argument("--cnn-checkpoint", default=None,
                   help="stage (a) classifier checkpoint to warm-start the encoder from")
    p.add_argument("--resume", default=None, help="report-generator checkpoint to resume")
    p.add_argument("--unfreeze-cnn", action="store_true",
                   help="phase 2: unfreeze denseblock4 + norm5")
    p.add_argument("--eval-batches", type=int, default=0, help="0 = whole val split")
    p.add_argument("--gpu-fraction", type=float, default=0.8)
    p.add_argument("--out", default=str(CKPT / "report_generator.pt"))
    args = p.parse_args()

    device = get_device()
    print(f"device: {device}")
    cap_gpu_memory(args.gpu_fraction, device)
    keep_awake(True)

    print("\nloading data...")
    train_loader, val_loader, tokenizer = make_report_loaders(
        batch_size=args.batch_size, max_len=args.max_len, min_freq=args.min_freq,
        num_workers=args.num_workers, clahe=args.clahe, limit=args.overfit or None)
    print(f"  vocab {tokenizer.vocab_size} tokens | "
          f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)}")
    if args.overfit:
        print(f"  OVERFIT MODE: {args.overfit} examples, augmentation off, "
              f"val == train. Loss should approach 0.")

    print("\nbuilding model...")
    model = build_model(args, tokenizer, device)
    total, trainable = count_parameters(model)
    print(f"  params {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"  resumed from {args.resume} (epoch {start_epoch}, "
              f"val loss {ckpt.get('val_loss', float('nan')):.4f})")

    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr, args.cnn_lr), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    max_iters = args.epochs * steps_per_epoch
    warmup_iters = int(args.warmup_frac * max_iters)
    stopper = EarlyStopping(patience=args.patience, mode="min", min_delta=1e-3)

    print(f"\ntraining {args.epochs} epochs x {steps_per_epoch} steps = {max_iters} steps")
    start = time.time()
    it = 0
    best_val = float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_loss = AverageMeter()
        for images, tgt_in, tgt_out in train_loader:
            lr = cosine_lr(it, args.lr, warmup_iters, max_iters)
            # Keep the CNN group at its own (much smaller) LR on the same schedule shape.
            optimizer.param_groups[0]["lr"] = lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = lr * (args.cnn_lr / args.lr)

            images = images.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                _, loss = model(images, tgt_in, tgt_out)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.update(loss.item(), images.size(0))
            it += 1

        val_loss = estimate_loss(model, val_loader, device, args.eval_batches or None)
        elapsed = format_seconds(time.time() - start)
        print(f"\nepoch {epoch + 1}  train {epoch_loss.avg:.4f}  val {val_loss:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  [{elapsed}]")
        sample_reports(model, val_loader, tokenizer, device)

        improved, should_stop = stopper.step(val_loss)
        if improved:
            best_val = val_loss
            save_checkpoint({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "vocab": tokenizer.itos,
                "config": {
                    "n_emb": args.n_emb, "n_head": args.n_head, "n_layer": args.n_layer,
                    "block_size": args.block_size, "dropout": args.dropout,
                    "factorized_pos": args.factorized_pos,
                    "vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id,
                },
                "args": vars(args),
            }, args.out)
            print(f"    saved (best val {val_loss:.4f}) -> {args.out}")
        if should_stop:
            print(f"\nearly stop: no improvement for {args.patience} epochs")
            break

    keep_awake(False)
    print(f"\ndone in {format_seconds(time.time() - start)}. best val loss {best_val:.4f}")
    print(f"Next: python evaluate.py --checkpoint {args.out}")


if __name__ == "__main__":
    main()
