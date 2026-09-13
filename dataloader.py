"""Image loading, report tokenisation, and the (image, report) paired dataset.

Covers PROJECT_PLAN.md Section 2 (image preprocessing) and the text half of Section 3
(word-level vocab with a frequency cutoff).

Two dataset flavours share the same image pipeline:
  * `CXRClassificationDataset` -> (image, label_vec)          for stage (a)
  * `CXRReportDataset`         -> (image, tgt_in, tgt_out)    for stage (c)
"""

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from labels import CONDITIONS

IGNORE_INDEX = -100
IMAGE_SIZE = 224
RESIZE_SIZE = 256  # decode larger, random-crop to 224 for train-time augmentation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT = Path(__file__).resolve().parent
# Overridable so the dataset can live outside a cloud-synced folder - see prepare_data.py.
DATA = Path(os.environ.get("CXR_DATA_DIR") or (ROOT / "data"))
MANIFEST = DATA / "processed" / "manifest.csv"
VOCAB_PATH = DATA / "processed" / "vocab.json"

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Word-level tokenisation: keep alphanumeric runs and sentence punctuation, drop the rest.
# Radiology reports are templated enough that this beats subwords at this data scale.
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[.,;:]")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------

class ReportTokenizer:
    """Word-level vocab built from the *training* reports only, with a frequency cutoff.

    Plan Section 3: "drop tokens seen <3 times" - radiology reports are far more
    repetitive than literary text, so this lands in the low thousands of tokens rather
    than the ~100k of a BPE vocab, which is what kills the B*T*vocab_size logits blowup.
    """

    # Set by `build`/`load`; defaults here so a directly-constructed tokenizer (e.g. from a
    # checkpoint's stored vocab) still has the attributes.
    fingerprint = None
    coverage = None

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self):
        return len(self.itos)

    @property
    def vocab_size(self):
        return len(self.itos)

    @classmethod
    def build(cls, texts, min_freq=3):
        counts = Counter()
        for t in texts:
            counts.update(tokenize(t))
        kept = sorted(
            (tok for tok, n in counts.items() if n >= min_freq),
            key=lambda t: (-counts[t], t),
        )
        itos = SPECIALS + kept
        tok = cls(itos)
        tok.coverage = _coverage(counts, set(kept))
        return tok

    def encode(self, text, add_special=True):
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if skip_special:
                if tok == self.itos[self.eos_id]:
                    break
                if tok in (PAD_TOKEN, BOS_TOKEN):
                    continue
            out.append(tok)
        return detokenize(out)

    def save(self, path=VOCAB_PATH, fingerprint=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "fingerprint": fingerprint}, f)
        return path

    @classmethod
    def load(cls, path=VOCAB_PATH):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        tok = cls(blob["itos"])
        tok.fingerprint = blob.get("fingerprint")
        return tok


def detokenize(tokens):
    """Join word tokens back into readable text (no space before punctuation)."""
    out = ""
    for tok in tokens:
        if tok in {".", ",", ";", ":"} or not out:
            out += tok
        else:
            out += " " + tok
    return out.strip()


def _coverage(counts, kept):
    total = sum(counts.values())
    covered = sum(n for tok, n in counts.items() if tok in kept)
    return {
        "types_total": len(counts),
        "types_kept": len(kept),
        "token_coverage": round(covered / max(1, total), 4),
    }


# --------------------------------------------------------------------------------------
# Image pipeline (plan Section 2)
# --------------------------------------------------------------------------------------

def _clahe(img):
    """Optional CLAHE contrast enhancement (plan Section 2.7). No-op without opencv."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return img
    arr = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(arr))


def build_transform(train, clahe=False):
    """Grayscale -> 3-channel -> ImageNet normalisation, with train-only augmentation.

    No horizontal flip: the cardiac silhouette and organ situs are laterality-dependent,
    so flipping teaches wrong anatomy (plan Section 2.5). No hue/saturation jitter either -
    meaningless on a grayscale image.
    """
    pre = [transforms.Grayscale(num_output_channels=1)]
    if clahe:
        pre.append(transforms.Lambda(_clahe))

    if train:
        geom = [
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    else:
        geom = [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ]

    return transforms.Compose(pre + geom + [
        transforms.Grayscale(num_output_channels=3),  # replicate the single channel 3x
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(path):
    """Open a CXR as a single-channel PIL image.

    IU X-Ray ships PNG. MIMIC-CXR ships DICOM - handled here so the same dataset class
    works once PhysioNet credentialing comes through (plan Section 2.1).
    """
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return _load_dicom(path)
    return Image.open(path).convert("L")


def _load_dicom(path):
    import numpy as np
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype("float32")
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    # MONOCHROME1 means "low value = white" - invert so bone is bright like every PNG.
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) if hi > lo else arr * 0
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------------------
# Manifest + datasets
# --------------------------------------------------------------------------------------

def read_manifest(path=MANIFEST):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python prepare_data.py` first (plan week 1)."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_rows(rows, split):
    return [r for r in rows if r["split"] == split]


def label_vector(row):
    return [float(row[f"lbl_{c.replace(' ', '_')}"]) for c in CONDITIONS]


class _BaseCXRDataset(Dataset):
    def __init__(self, rows, train, data_root=DATA, clahe=False):
        self.rows = rows
        self.data_root = Path(data_root)
        self.transform = build_transform(train=train, clahe=clahe)

    def __len__(self):
        return len(self.rows)

    def _image(self, row):
        img = load_image(self.data_root / row["image_path"])
        return self.transform(img)


class CXRClassificationDataset(_BaseCXRDataset):
    """Stage (a): image -> 14-way multi-label target."""

    def __getitem__(self, i):
        row = self.rows[i]
        return self._image(row), torch.tensor(label_vector(row), dtype=torch.float32)


class CXRReportDataset(_BaseCXRDataset):
    """Stage (c): image -> report token ids (variable length, padded by the collator)."""

    def __init__(self, rows, tokenizer, train, max_len=160, data_root=DATA, clahe=False):
        super().__init__(rows, train=train, data_root=data_root, clahe=clahe)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, i):
        row = self.rows[i]
        ids = self.tokenizer.encode(row["report"])[: self.max_len + 1]
        return self._image(row), torch.tensor(ids, dtype=torch.long)


def make_collate_fn(pad_id):
    """Pad a batch to its own longest sequence and build the shifted in/out pair.

    Padding to the batch max rather than a fixed block_size is the one real departure
    from the translation project: report lengths here are very skewed (median ~35 words,
    p95 ~90), so fixed-length padding would waste most of every batch.
    """

    def collate(batch):
        images, seqs = zip(*batch)
        images = torch.stack(images)
        longest = max(len(s) for s in seqs)

        tgt_in = torch.full((len(seqs), longest - 1), pad_id, dtype=torch.long)
        tgt_out = torch.full((len(seqs), longest - 1), IGNORE_INDEX, dtype=torch.long)
        for i, s in enumerate(seqs):
            n = len(s) - 1
            tgt_in[i, :n] = s[:-1]
            tgt_out[i, :n] = s[1:]
        return images, tgt_in, tgt_out

    return collate


def _vocab_fingerprint(train_reports, min_freq):
    """Identifies the corpus a cached vocab was built from."""
    h = hashlib.md5(f"{min_freq}|{len(train_reports)}|".encode())
    for r in train_reports:
        h.update(r.encode("utf-8", "replace"))
    return h.hexdigest()


def build_tokenizer(rows=None, min_freq=3, rebuild=False, path=VOCAB_PATH):
    """Load the cached vocab, or build it from the training split and cache it.

    The cache is fingerprinted against the training reports it came from. Without that, a
    vocab left over from a different manifest (a re-split, a subset, a synthetic test
    fixture) would be loaded silently and the model would train against the wrong token
    ids - a failure that shows up only as mysteriously bad output much later.
    """
    path = Path(path)
    if rows is None:
        rows = read_manifest()
    train_reports = [r["report"] for r in split_rows(rows, "train")]
    fingerprint = _vocab_fingerprint(train_reports, min_freq)

    if path.exists() and not rebuild:
        cached = ReportTokenizer.load(path)
        if cached.fingerprint == fingerprint:
            return cached
        print(f"  cached vocab at {path} does not match the current training split "
              f"(or min_freq); rebuilding")

    tok = ReportTokenizer.build(train_reports, min_freq=min_freq)
    tok.save(path, fingerprint=fingerprint)
    print(f"built vocab: {len(tok)} tokens from {len(train_reports)} training reports "
          f"(min_freq={min_freq}, token coverage {tok.coverage['token_coverage']:.3%})")
    return tok


def make_report_loaders(batch_size=16, max_len=160, min_freq=3, num_workers=4,
                        clahe=False, limit=None, rows=None, data_root=DATA,
                        vocab_path=VOCAB_PATH):
    """Train/val DataLoaders for stage (c), plus the tokenizer.

    `limit` truncates the training split - that is the tiny-overfit-subset debugging
    harness from plan weeks 4-5 (50-200 pairs until the model memorises them).
    """
    rows = rows if rows is not None else read_manifest()
    tok = build_tokenizer(rows, min_freq=min_freq, path=vocab_path)

    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    if limit:
        train_rows = train_rows[:limit]
        val_rows = train_rows  # overfit mode: validate on the same handful of examples

    collate = make_collate_fn(tok.pad_id)
    common = dict(batch_size=batch_size, collate_fn=collate, num_workers=num_workers,
                  pin_memory=True, persistent_workers=num_workers > 0)
    train_ds = CXRReportDataset(train_rows, tok, train=not limit, max_len=max_len,
                                clahe=clahe, data_root=data_root)
    val_ds = CXRReportDataset(val_rows, tok, train=False, max_len=max_len,
                              clahe=clahe, data_root=data_root)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, tok


def make_classification_loaders(batch_size=32, num_workers=4, clahe=False, rows=None,
                                data_root=DATA):
    """Train/val DataLoaders for stage (a)."""
    rows = rows if rows is not None else read_manifest()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_ds = CXRClassificationDataset(split_rows(rows, "train"), train=True,
                                        clahe=clahe, data_root=data_root)
    val_ds = CXRClassificationDataset(split_rows(rows, "val"), train=False,
                                      clahe=clahe, data_root=data_root)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )


if __name__ == "__main__":
    rows = read_manifest()
    print(f"manifest: {len(rows)} images, {len({r['uid'] for r in rows})} studies")
    tok = build_tokenizer(rows, rebuild=True)
    print(f"vocab size: {len(tok)}")
    print("most common:", tok.itos[4:24])

    train_loader, val_loader, tok = make_report_loaders(batch_size=4, num_workers=0, limit=8)
    images, tgt_in, tgt_out = next(iter(train_loader))
    print(f"images {tuple(images.shape)}  tgt_in {tuple(tgt_in.shape)}  "
          f"tgt_out {tuple(tgt_out.shape)}")
    print("roundtrip:", tok.decode(tgt_in[0].tolist())[:160])
