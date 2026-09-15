"""Report generation CLI - the equivalent of `transformer/translate.py`.

    python generate.py --image data/raw/images/CXR1_1_IM-0001-3001.png
    python generate.py --image path/to/study.dcm --beam-size 5 --show-labels
    python generate.py --split test --limit 5        # sample from the held-out split
"""

import argparse
from pathlib import Path

import torch

from dataloader import build_transform, load_image, read_manifest, split_rows
from evaluate import load_generator
from labels import positives, labels_from_text
from utils import get_device


def prepare_image(path, clahe=False):
    """One image file -> (1, 3, 224, 224) tensor, using the eval-time transform."""
    transform = build_transform(train=False, clahe=clahe)
    return transform(load_image(path)).unsqueeze(0)


@torch.no_grad()
def generate_report(model, tokenizer, image_tensor, device, beam_size=3,
                    max_new_tokens=120, length_penalty=0.6):
    image_tensor = image_tensor.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type == "cuda")):
        out = model.generate(image_tensor, max_new_tokens, tokenizer.bos_id,
                             tokenizer.eos_id, beam_size=beam_size,
                             length_penalty=length_penalty)
    return [tokenizer.decode(seq.tolist()) for seq in out]


def main():
    p = argparse.ArgumentParser(description="Generate a radiology report from a chest X-ray.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a PNG/JPG/DICOM chest X-ray")
    src.add_argument("--split", choices=["train", "val", "test"],
                     help="sample images from a manifest split instead")
    p.add_argument("--checkpoint", default="checkpoints/rg_v5a_phase2.pt")
    p.add_argument("--beam-size", type=int, default=1,
                   help="1 = greedy (default: beam search drifts to the normal template)")
    p.add_argument("--length-penalty", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--limit", type=int, default=5, help="how many to sample with --split")
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--show-labels", action="store_true",
                   help="also print the conditions the rule-based labeller finds")
    args = p.parse_args()

    device = get_device()
    model, tokenizer = load_generator(args.checkpoint, device)

    if args.image:
        path = Path(args.image)
        if not path.exists():
            raise SystemExit(f"no such image: {path}")
        report = generate_report(model, tokenizer, prepare_image(path, args.clahe), device,
                                 args.beam_size, args.max_new_tokens, args.length_penalty)[0]
        print(f"\n{path.name}\n{report}")
        if args.show_labels:
            print(f"\nconditions: {positives(labels_from_text(report)) or ['(none detected)']}")
        return

    rows = split_rows(read_manifest(), args.split)[:args.limit]
    data_root = Path(__file__).resolve().parent / "data"
    for row in rows:
        image = prepare_image(data_root / row["image_path"], args.clahe)
        report = generate_report(model, tokenizer, image, device, args.beam_size,
                                 args.max_new_tokens, args.length_penalty)[0]
        print(f"\n=== {row['uid']} ({row['image_id']}) ===")
        print(f"  [reference] {row['report']}")
        print(f"  [generated] {report}")
        if args.show_labels:
            print(f"  [ref labels] {positives(labels_from_text(row['report']))}")
            print(f"  [gen labels] {positives(labels_from_text(report))}")


if __name__ == "__main__":
    main()
