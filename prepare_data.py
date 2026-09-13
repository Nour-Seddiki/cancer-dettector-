"""Week 1: download IU X-Ray, parse the reports, build a manifest, split by study.

IU X-Ray (Indiana University Chest X-Ray Collection) is distributed by Open-i as two
tarballs - the PNG images and the ecgen-radiology XML reports. Both are public, no
credentialing (PROJECT_PLAN.md Section 0).

Usage:
    python prepare_data.py                 # download + extract + build manifest
    python prepare_data.py --skip-download # reuse data/raw, just rebuild the manifest

Output: data/processed/manifest.csv, one row per *image*, with the study-level report and
labels denormalised onto it, plus a `split` column assigned by study so an image and its
lateral counterpart never straddle the train/test boundary (the leakage trap in the plan).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from labels import CONDITIONS, label_study, labels_from_mesh

IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"

ROOT = Path(__file__).resolve().parent
# The default `data/` lives under the project, which on this machine is inside OneDrive -
# it will try to sync ~1.4 GB of tarball plus 7,470 PNGs. Point CXR_DATA_DIR somewhere
# outside the synced tree (e.g. C:\ml-data\iu-xray) to avoid that.
DATA = Path(os.environ.get("CXR_DATA_DIR") or (ROOT / "data"))
RAW = DATA / "raw"
IMAGE_DIR = RAW / "images"
REPORT_DIR = RAW / "reports"
PROCESSED = DATA / "processed"
MANIFEST = PROCESSED / "manifest.csv"

# IU X-Ray anonymises PHI by substituting runs of X. Also strips de-identification
# artefacts like "XXXX-year-old" that would otherwise dominate the vocabulary.
ANON = re.compile(r"\bX{2,}\b")
WS = re.compile(r"\s+")


def download(url, dest, attempts=8, backoff=5.0):
    """Stream a URL to disk, resuming with HTTP Range across dropped connections.

    Open-i resets the connection partway through the 1.4 GB image tarball often enough
    that a non-resuming download effectively never finishes. The partial file is kept
    between attempts and re-requested with a `Range: bytes=N-` header, so each retry picks
    up where the last one died instead of starting over.
    """
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {dest.name} already present ({dest.stat().st_size / 1e6:.0f} MB), skipping")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")

    total = None
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as r:
                # 416 = we already have the whole file; the server has nothing left to send.
                if r.status_code == 416 and have:
                    break
                # A server ignoring Range replies 200 and restarts the body at byte 0.
                if have and r.status_code != 206:
                    print(f"\n    server ignored Range (HTTP {r.status_code}); restarting")
                    tmp.unlink(missing_ok=True)
                    have = 0
                r.raise_for_status()

                length = int(r.headers.get("content-length", 0)) or None
                if length:
                    total = have + length
                done = have
                mode = "ab" if have else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r    {done / 1e6:7.0f} / {total / 1e6:.0f} MB "
                                  f"({100 * done / total:5.1f}%)", end="", flush=True)
                        else:
                            print(f"\r    {done / 1e6:7.0f} MB", end="", flush=True)
            print()
            if total is None or tmp.stat().st_size >= total:
                break
            print(f"    short read ({tmp.stat().st_size / 1e6:.0f} of {total / 1e6:.0f} MB)")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # connection reset, timeout, chunked-encoding break
            got = tmp.stat().st_size if tmp.exists() else 0
            print(f"\n    attempt {attempt}/{attempts} failed at {got / 1e6:.0f} MB: "
                  f"{type(e).__name__}")
            if attempt == attempts:
                raise
            wait = backoff * attempt
            print(f"    retrying in {wait:.0f}s (resuming from {got / 1e6:.0f} MB)")
            time.sleep(wait)

    size = tmp.stat().st_size if tmp.exists() else 0
    if total and size < total:
        raise RuntimeError(f"{dest.name}: got {size} of {total} bytes after {attempts} attempts")
    if size == 0:
        raise RuntimeError(f"{dest.name}: downloaded nothing")
    tmp.replace(dest)
    print(f"  saved {dest.name} ({size / 1e6:.0f} MB)")
    return dest


def extract(tgz_path, dest_dir, expect_ext):
    """Extract a tarball flat into dest_dir, skipping if it already looks populated."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for p in dest_dir.iterdir() if p.suffix.lower() == expect_ext)
    if existing > 100:
        print(f"  {dest_dir.name}/ already has {existing} {expect_ext} files, skipping extract")
        return

    print(f"  extracting {tgz_path.name} -> {dest_dir}")
    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name or Path(name).suffix.lower() != expect_ext:
                continue
            target = dest_dir / name
            if target.exists():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                out.write(src.read())
    print(f"  extracted {sum(1 for _ in dest_dir.iterdir())} files")


def clean_text(s):
    """Normalise a report field: collapse whitespace, drop XXXX anonymisation tokens."""
    if not s:
        return ""
    s = ANON.sub(" ", s)
    s = WS.sub(" ", s).strip()
    # A field that was *only* anonymisation tokens is worthless.
    if s in {".", "..", "-", "None.", "None"}:
        return ""
    return s


def parse_report(xml_path):
    """Parse one ecgen-radiology XML file into a study dict, or None if unusable."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    fields = {}
    for at in root.iter("AbstractText"):
        label = (at.get("Label") or "").strip().upper()
        if label:
            fields[label] = clean_text(at.text)

    findings = fields.get("FINDINGS", "")
    impression = fields.get("IMPRESSION", "")
    if not findings and not impression:
        return None

    # The generation target: FINDINGS then IMPRESSION, the convention in the IU X-Ray
    # report-generation literature (R2Gen and successors use the same concatenation).
    report = " ".join(p for p in (findings, impression) if p).strip()
    if len(report.split()) < 3:
        return None

    mesh_major = [e.text.strip() for e in root.iter("major") if e.text and e.text.strip()]
    mesh_minor = [e.text.strip() for e in root.iter("automatic") if e.text and e.text.strip()]
    mesh = mesh_major + mesh_minor

    uid_el = root.find("uId")
    uid = (uid_el.get("id") if uid_el is not None else None) or xml_path.stem

    images = []
    for pi in root.iter("parentImage"):
        img_id = pi.get("id")
        if img_id:
            images.append(img_id)

    return {
        "uid": uid,
        "report_file": xml_path.name,
        "findings": findings,
        "impression": impression,
        "report": report,
        "indication": fields.get("INDICATION", ""),
        "comparison": fields.get("COMPARISON", ""),
        "mesh": mesh,
        "images": sorted(images),
    }


def assign_view(image_ids):
    """Heuristic frontal/lateral assignment for a study.

    The PNG release carries no view metadata (it is in the DICOM headers, which Open-i
    does not ship). IU X-Ray studies are acquired as a PA/AP frontal followed by a
    lateral, and the parentImage ids preserve that acquisition order, so the
    lowest-sorting id of a study is taken as the frontal view. This is the same
    images[0]/images[1] convention R2Gen uses.

    It is a heuristic, not ground truth - `--all-views` keeps every image if you would
    rather not rely on it, and view is recorded in the manifest either way.
    """
    views = {}
    for i, img_id in enumerate(sorted(image_ids)):
        views[img_id] = "frontal" if i == 0 else "lateral"
    return views


def stable_split(uid, val_frac, test_frac, seed):
    """Deterministic per-study split - hash-based, so it survives rebuilds and reorderings."""
    h = hashlib.md5(f"{seed}:{uid}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / float(1 << 64)
    if x < test_frac:
        return "test"
    if x < test_frac + val_frac:
        return "val"
    return "train"


def build_manifest(args):
    xml_files = sorted(REPORT_DIR.glob("*.xml"))
    if not xml_files:
        sys.exit(f"No XML reports found in {REPORT_DIR}. Run without --skip-download first.")
    print(f"  parsing {len(xml_files)} report XML files")

    available = {p.stem: p for p in IMAGE_DIR.glob("*.png")}
    print(f"  found {len(available)} PNG images")
    if not available:
        sys.exit(f"No PNG images found in {IMAGE_DIR}. Run without --skip-download first.")

    rows = []
    stats = Counter()
    for xml_path in xml_files:
        study = parse_report(xml_path)
        if study is None:
            stats["studies_dropped_no_report"] += 1
            continue

        present = [i for i in study["images"] if i in available]
        if not present:
            stats["studies_dropped_no_image"] += 1
            continue

        views = assign_view(present)
        mesh_vec = labels_from_mesh(study["mesh"])
        label_vec = label_study(study["mesh"], study["report"])
        label_source = "mesh" if mesh_vec is not None else "text"
        stats[f"label_source_{label_source}"] += 1

        split = stable_split(study["uid"], args.val_frac, args.test_frac, args.seed)
        stats[f"studies_{split}"] += 1
        stats["studies_kept"] += 1

        for img_id in sorted(present):
            view = views[img_id]
            if not args.all_views and view != "frontal":
                stats["images_skipped_non_frontal"] += 1
                continue
            stats["images_kept"] += 1
            rows.append({
                "uid": study["uid"],
                "image_id": img_id,
                "image_path": str(Path("raw/images") / f"{img_id}.png").replace("\\", "/"),
                "view": view,
                "split": split,
                "findings": study["findings"],
                "impression": study["impression"],
                "report": study["report"],
                "indication": study["indication"],
                "mesh": "|".join(study["mesh"]),
                "label_source": label_source,
                **{f"lbl_{c.replace(' ', '_')}": v for c, v in zip(CONDITIONS, label_vec)},
            })

    if not rows:
        sys.exit("Manifest is empty - nothing matched. Check the raw data directories.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows, stats, args)
    with open(PROCESSED / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  wrote {MANIFEST} ({len(rows)} rows)")
    return summary


def summarise(rows, stats, args):
    by_split = Counter(r["split"] for r in rows)
    studies_by_split = {
        s: len({r["uid"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")
    }
    lengths = sorted(len(r["report"].split()) for r in rows)
    pos = {c: sum(int(r[f"lbl_{c.replace(' ', '_')}"]) for r in rows) for c in CONDITIONS}

    def pct(k):
        return round(100 * k / max(1, len(rows)), 1)

    summary = {
        "images": len(rows),
        "studies": len({r["uid"] for r in rows}),
        "images_by_split": dict(by_split),
        "studies_by_split": studies_by_split,
        "views": dict(Counter(r["view"] for r in rows)),
        "report_words": {
            "min": lengths[0],
            "p25": lengths[len(lengths) // 4],
            "median": lengths[len(lengths) // 2],
            "p75": lengths[3 * len(lengths) // 4],
            "p95": lengths[int(0.95 * (len(lengths) - 1))],
            "max": lengths[-1],
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "label_prevalence": {c: {"n": n, "pct": pct(n)} for c, n in pos.items()},
        "parse_stats": dict(stats),
        "config": {
            "val_frac": args.val_frac, "test_frac": args.test_frac,
            "seed": args.seed, "all_views": args.all_views,
        },
    }

    print("\n=== IU X-Ray manifest ===")
    print(f"  studies kept   : {summary['studies']}   images: {summary['images']}")
    print(f"  split (studies): {studies_by_split}")
    print(f"  split (images) : {dict(by_split)}")
    print(f"  views          : {summary['views']}")
    rw = summary["report_words"]
    print(f"  report length  : median {rw['median']} words, p95 {rw['p95']}, max {rw['max']}")
    print("  label prevalence:")
    for c in CONDITIONS:
        print(f"    {c:28s} {pos[c]:5d}  ({pct(pos[c]):4.1f}%)")
    print(f"  label source   : mesh={stats['label_source_mesh']} text={stats['label_source_text']}")
    dropped = stats["studies_dropped_no_report"] + stats["studies_dropped_no_image"]
    print(f"  dropped studies: {dropped} "
          f"(no report: {stats['studies_dropped_no_report']}, "
          f"no image: {stats['studies_dropped_no_image']})")
    return summary


def main():
    p = argparse.ArgumentParser(description="Download and prepare the IU X-Ray dataset.")
    p.add_argument("--skip-download", action="store_true",
                   help="reuse whatever is already in data/raw")
    p.add_argument("--all-views", action="store_true",
                   help="keep lateral views too (default: frontal only, per plan Section 2.6)")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    if not args.skip_download:
        print("[1/3] downloading")
        RAW.mkdir(parents=True, exist_ok=True)
        img_tgz = download(IMAGES_URL, RAW / "NLMCXR_png.tgz")
        rep_tgz = download(REPORTS_URL, RAW / "NLMCXR_reports.tgz")
        print("[2/3] extracting")
        extract(img_tgz, IMAGE_DIR, ".png")
        extract(rep_tgz, REPORT_DIR, ".xml")
    else:
        print("[1-2/3] skipping download/extract")

    print("[3/3] building manifest")
    build_manifest(args)
    print("\nNext: python eda.py  (then train_classifier.py)")


if __name__ == "__main__":
    main()
