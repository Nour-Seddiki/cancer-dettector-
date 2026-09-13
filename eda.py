"""Week 1 EDA: report lengths, vocabulary, view counts, class imbalance.

    python eda.py            # print the summary
    python eda.py --plots    # also write outputs/eda_*.png

The numbers worth looking at before training anything:
  * how many reports are boilerplate-normal (sets the floor BLEU a trivial model reaches)
  * report length p95 (sets `--max-len`)
  * vocab size at min_freq=3 (sets the softmax width)
  * per-condition prevalence (tells you which AUROCs will be meaningless)
"""

import argparse
from collections import Counter
from pathlib import Path

from dataloader import ReportTokenizer, read_manifest, split_rows, tokenize
from labels import CONDITIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))]


def main():
    p = argparse.ArgumentParser(description="EDA over the IU X-Ray manifest.")
    p.add_argument("--plots", action="store_true", help="write PNG plots to outputs/")
    p.add_argument("--min-freq", type=int, default=3)
    args = p.parse_args()

    rows = read_manifest()
    studies = {r["uid"] for r in rows}
    print(f"=== dataset ===")
    print(f"  images  {len(rows)}")
    print(f"  studies {len(studies)}")
    print(f"  views   {dict(Counter(r['view'] for r in rows))}")
    for split in ("train", "val", "test"):
        sr = split_rows(rows, split)
        print(f"  {split:5s}  {len(sr):5d} images  {len({r['uid'] for r in sr}):5d} studies")

    # -- report length ------------------------------------------------------------------
    lengths = sorted(len(tokenize(r["report"])) for r in rows)
    print(f"\n=== report length (word tokens) ===")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"  p{int(q * 100):02d}  {percentile(lengths, q)}")
    print(f"  min {lengths[0]}  max {lengths[-1]}  mean {sum(lengths) / len(lengths):.1f}")
    print(f"  -> a --max-len of {percentile(lengths, 0.99) + 2} covers 99% of reports")

    # -- vocabulary ---------------------------------------------------------------------
    train_reports = [r["report"] for r in split_rows(rows, "train")]
    counts = Counter()
    for t in train_reports:
        counts.update(tokenize(t))
    tok = ReportTokenizer.build(train_reports, min_freq=args.min_freq)
    print(f"\n=== vocabulary (train split) ===")
    print(f"  types (all)           {len(counts)}")
    print(f"  types (min_freq>={args.min_freq})   {len(tok) - 4}")
    print(f"  token coverage        {tok.coverage['token_coverage']:.3%}")
    print(f"  vocab_size (+ 4 specials) {len(tok)}")
    print(f"  top 25: {', '.join(tok.itos[4:29])}")
    hapax = sum(1 for c in counts.values() if c == 1)
    print(f"  hapax legomena        {hapax} ({100 * hapax / len(counts):.1f}% of types)")

    # -- class imbalance ----------------------------------------------------------------
    print(f"\n=== condition prevalence (per image) ===")
    n = len(rows)
    for c in CONDITIONS:
        k = sum(int(r[f"lbl_{c.replace(' ', '_')}"]) for r in rows)
        bar = "#" * int(40 * k / max(1, n))
        print(f"  {c:28s} {k:5d} ({100 * k / n:5.1f}%) {bar}")
    sources = Counter(r["label_source"] for r in rows)
    print(f"  label source: {dict(sources)}")

    # -- boilerplate --------------------------------------------------------------------
    normalised = Counter(" ".join(tokenize(r["report"])) for r in rows)
    top = normalised.most_common(5)
    print(f"\n=== most common reports verbatim ===")
    for text, k in top:
        print(f"  {k:4d}x ({100 * k / n:4.1f}%)  {text[:110]}")
    print(f"  distinct reports: {len(normalised)} / {n} images "
          f"({100 * len(normalised) / n:.1f}%)")
    print("  -> a model that always emits the single most common report already scores")
    print(f"     a non-trivial BLEU. Judge against that floor, not against zero.")

    if args.plots:
        make_plots(lengths, counts, rows)


def make_plots(lengths, counts, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping plots")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=60, color="#3b6ea5")
    ax.axvline(percentile(lengths, 0.95), color="#c0392b", ls="--",
               label=f"p95 = {percentile(lengths, 0.95)}")
    ax.set_xlabel("report length (word tokens)")
    ax.set_ylabel("images")
    ax.set_title("IU X-Ray report length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "eda_report_length.png", dpi=130)
    plt.close(fig)

    freqs = sorted(counts.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(range(1, len(freqs) + 1), freqs, color="#3b6ea5")
    ax.set_xlabel("rank")
    ax.set_ylabel("frequency")
    ax.set_title("Vocabulary frequency (Zipf)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_vocab_zipf.png", dpi=130)
    plt.close(fig)

    n = len(rows)
    prev = [(c, sum(int(r[f"lbl_{c.replace(' ', '_')}"]) for r in rows) / n) for c in CONDITIONS]
    prev.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([c for c, _ in prev], [100 * v for _, v in prev], color="#3b6ea5")
    ax.set_xlabel("% of images")
    ax.set_title("Condition prevalence")
    fig.tight_layout()
    fig.savefig(OUT / "eda_prevalence.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote plots to {OUT}/")


if __name__ == "__main__":
    main()
