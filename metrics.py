"""Week 6 metrics: BLEU-1..4, ROUGE-L, CIDEr-D, and clinical efficacy.

The NLG metrics are implemented here rather than taken as a hard dependency, because
`pycocoevalcap` pulls in a Java toolchain for METEOR/SPICE that this project never uses.
When `pycocoevalcap` *is* installed, `nlg_metrics(..., prefer_coco=True)` defers to it so
the numbers are directly comparable to published results.

These implementations follow the pycocoevalcap ones closely, including CIDEr-D's
length-penalty quirk (its "length" is the bigram count, not the word count) so that scores
line up with the literature rather than being a private variant.

The metric that actually matters is the clinical one (plan Section 4): BLEU rewards a
model that emits "no acute cardiopulmonary abnormality" for every study, because most
reports really do say that.
"""

import math
from collections import Counter, defaultdict

from labels import CONDITIONS, NUM_CONDITIONS, labels_from_text

CIDER_N = 4
CIDER_SIGMA = 6.0
ROUGE_BETA = 1.2


# --------------------------------------------------------------------------------------
# n-gram helpers
# --------------------------------------------------------------------------------------

def _tokens(text):
    return text.lower().split()


def _ngrams(tokens, n_max=4):
    """Counter over all 1..n_max grams of a token list."""
    counts = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i:i + n])] += 1
    return counts


# --------------------------------------------------------------------------------------
# BLEU
# --------------------------------------------------------------------------------------

def bleu(hypotheses, references, max_n=4):
    """Corpus-level BLEU-1..max_n with the standard closest-reference brevity penalty.

    `references[i]` may be a string or a list of strings (multiple references).
    Returns {"BLEU-1": .., ..., "BLEU-4": ..}.
    """
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hypotheses, references):
        refs = [refs] if isinstance(refs, str) else refs
        h_toks = _tokens(hyp)
        r_tok_lists = [_tokens(r) for r in refs]

        hyp_len += len(h_toks)
        # brevity penalty uses the reference length closest to the hypothesis length
        ref_len += min((abs(len(r) - len(h_toks)), len(r)) for r in r_tok_lists)[1] \
            if r_tok_lists else 0

        h_counts = _ngrams(h_toks, max_n)
        max_ref_counts = Counter()
        for r_toks in r_tok_lists:
            for ng, c in _ngrams(r_toks, max_n).items():
                if c > max_ref_counts[ng]:
                    max_ref_counts[ng] = c

        for ng, c in h_counts.items():
            n = len(ng) - 1
            clipped[n] += min(c, max_ref_counts[ng])
        for n in range(max_n):
            totals[n] += max(0, len(h_toks) - n)

    precisions = []
    for n in range(max_n):
        # Smoothing so a single missing 4-gram does not zero the whole corpus score on a
        # small test set (BLEU's usual failure mode at this data scale).
        num, den = clipped[n], totals[n]
        precisions.append((num / den) if num > 0 and den > 0 else (1e-9 if den else 0.0))

    bp = 1.0 if hyp_len > ref_len else (math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0)

    scores = {}
    for n in range(1, max_n + 1):
        log_avg = sum(math.log(p) for p in precisions[:n]) / n if all(precisions[:n]) else -1e9
        scores[f"BLEU-{n}"] = bp * math.exp(log_avg) if log_avg > -1e8 else 0.0
    return scores


# --------------------------------------------------------------------------------------
# ROUGE-L
# --------------------------------------------------------------------------------------

def _lcs_length(a, b):
    """Length of the longest common subsequence, O(len(a) * len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b):
            curr.append(prev[j] + 1 if x == y else max(curr[j], prev[j + 1]))
        prev = curr
    return prev[-1]


def rouge_l(hypotheses, references, beta=ROUGE_BETA):
    """Mean sentence-level ROUGE-L F-measure (beta=1.2, recall-weighted, as in coco-caption)."""
    scores = []
    for hyp, refs in zip(hypotheses, references):
        refs = [refs] if isinstance(refs, str) else refs
        h = _tokens(hyp)
        best = 0.0
        for ref in refs:
            r = _tokens(ref)
            lcs = _lcs_length(h, r)
            if lcs == 0:
                continue
            prec = lcs / len(h)
            rec = lcs / len(r)
            best = max(best, ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec))
        scores.append(best)
    return {"ROUGE-L": sum(scores) / len(scores) if scores else 0.0}


# --------------------------------------------------------------------------------------
# CIDEr-D
# --------------------------------------------------------------------------------------

def _cider_vec(counts, doc_freq, log_ref_len):
    vec = [defaultdict(float) for _ in range(CIDER_N)]
    norm = [0.0] * CIDER_N
    length = 0
    for ng, tf in counts.items():
        df = math.log(max(1.0, doc_freq[ng]))
        n = len(ng) - 1
        if n >= CIDER_N:
            continue
        vec[n][ng] = tf * (log_ref_len - df)
        norm[n] += vec[n][ng] ** 2
        # The reference implementation measures "length" in bigrams; kept for comparability.
        if n == 1:
            length += tf
    return vec, [math.sqrt(x) for x in norm], length


def _cider_sim(v_h, n_h, l_h, v_r, n_r, l_r):
    delta = float(l_h - l_r)
    vals = [0.0] * CIDER_N
    for n in range(CIDER_N):
        for ng, c in v_h[n].items():
            vals[n] += min(c, v_r[n][ng]) * v_r[n][ng]
        if n_h[n] != 0 and n_r[n] != 0:
            vals[n] /= (n_h[n] * n_r[n])
        vals[n] *= math.exp(-(delta ** 2) / (2 * CIDER_SIGMA ** 2))
    return vals


def cider(hypotheses, references):
    """CIDEr-D. Document frequencies come from the reference corpus being scored.

    Note: with a single reference per study and a corpus this small, CIDEr is noisy - it
    was designed for 5+ references per image. Report it, but weight BLEU/ROUGE and above
    all the clinical metric more heavily.
    """
    ref_lists = [[r] if isinstance(r, str) else list(r) for r in references]
    hyp_counts = [_ngrams(_tokens(h)) for h in hypotheses]
    ref_counts = [[_ngrams(_tokens(r)) for r in refs] for refs in ref_lists]

    doc_freq = Counter()
    for refs in ref_counts:
        for ng in {ng for c in refs for ng in c}:
            doc_freq[ng] += 1

    log_ref_len = math.log(float(max(1, len(ref_counts))))
    scores = []
    for h_cnt, r_cnts in zip(hyp_counts, ref_counts):
        v_h, n_h, l_h = _cider_vec(h_cnt, doc_freq, log_ref_len)
        acc = [0.0] * CIDER_N
        for r_cnt in r_cnts:
            v_r, n_r, l_r = _cider_vec(r_cnt, doc_freq, log_ref_len)
            for n, val in enumerate(_cider_sim(v_h, n_h, l_h, v_r, n_r, l_r)):
                acc[n] += val
        score = (sum(acc) / CIDER_N) / max(1, len(r_cnts)) * 10.0
        scores.append(score)
    return {"CIDEr": sum(scores) / len(scores) if scores else 0.0}


# --------------------------------------------------------------------------------------
# Aggregate NLG metrics
# --------------------------------------------------------------------------------------

def nlg_metrics(hypotheses, references, prefer_coco=True):
    """BLEU-1..4 + ROUGE-L + CIDEr. Uses pycocoevalcap when available."""
    if prefer_coco:
        coco = _coco_metrics(hypotheses, references)
        if coco is not None:
            return coco
    out = {}
    out.update(bleu(hypotheses, references))
    out.update(rouge_l(hypotheses, references))
    out.update(cider(hypotheses, references))
    return out


def _coco_metrics(hypotheses, references):
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        return None

    gts = {}
    res = {}
    for i, (h, r) in enumerate(zip(hypotheses, references)):
        refs = [r] if isinstance(r, str) else list(r)
        gts[i] = [x.lower() for x in refs]
        res[i] = [h.lower()]

    out = {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    for n, s in enumerate(bleu_scores, start=1):
        out[f"BLEU-{n}"] = float(s)
    out["ROUGE-L"] = float(Rouge().compute_score(gts, res)[0])
    out["CIDEr"] = float(Cider().compute_score(gts, res)[0])
    return out


# --------------------------------------------------------------------------------------
# Clinical efficacy (plan Section 4 - the headline metric)
# --------------------------------------------------------------------------------------

def clinical_efficacy(hypotheses, references, labeller=labels_from_text):
    """Label generated and reference reports, then score the agreement.

    Returns micro-averaged and per-condition precision/recall/F1. `labeller` is swappable:
    pass a CheXbert-backed callable to get the standard CE numbers.

    "No Finding" is excluded from the micro average - it is the majority class and
    including it flatters a model that just predicts normal for everything, which is the
    exact failure this metric exists to catch.
    """
    hyp_labels = [labeller(h) for h in hypotheses]
    ref_labels = [labeller(r if isinstance(r, str) else r[0]) for r in references]

    tp = [0] * NUM_CONDITIONS
    fp = [0] * NUM_CONDITIONS
    fn = [0] * NUM_CONDITIONS
    for hl, rl in zip(hyp_labels, ref_labels):
        for c in range(NUM_CONDITIONS):
            if hl[c] and rl[c]:
                tp[c] += 1
            elif hl[c] and not rl[c]:
                fp[c] += 1
            elif not hl[c] and rl[c]:
                fn[c] += 1

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    per_condition = {}
    for c, name in enumerate(CONDITIONS):
        p, r, f = prf(tp[c], fp[c], fn[c])
        per_condition[name] = {
            "precision": p, "recall": r, "f1": f,
            "support": tp[c] + fn[c], "tp": tp[c], "fp": fp[c], "fn": fn[c],
        }

    idx = [c for c in range(NUM_CONDITIONS) if CONDITIONS[c] != "No Finding"]
    mp, mr, mf = prf(sum(tp[c] for c in idx), sum(fp[c] for c in idx), sum(fn[c] for c in idx))

    scored = [c for c in idx if (tp[c] + fn[c]) > 0]
    macro_f1 = sum(per_condition[CONDITIONS[c]]["f1"] for c in scored) / len(scored) \
        if scored else 0.0

    return {
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "macro_f1": macro_f1,
        "per_condition": per_condition,
        "n_examples": len(hypotheses),
    }


# --------------------------------------------------------------------------------------
# Degeneration diagnostics (plan week 6 error analysis)
# --------------------------------------------------------------------------------------

def diversity_stats(hypotheses):
    """Catch the classic captioning failure modes: repetition and mode collapse."""
    if not hypotheses:
        return {}
    all_toks = [_tokens(h) for h in hypotheses]
    total_toks = sum(len(t) for t in all_toks)
    unique_reports = len({h.strip().lower() for h in hypotheses})

    # fraction of each report's 4-grams that are duplicates within that report
    rep_fracs = []
    for toks in all_toks:
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        if grams:
            rep_fracs.append(1 - len(set(grams)) / len(grams))

    return {
        "unique_reports": unique_reports,
        "unique_report_frac": unique_reports / len(hypotheses),
        "distinct_1": len({t for toks in all_toks for t in toks}) / max(1, total_toks),
        "distinct_2": len({tuple(toks[i:i + 2]) for toks in all_toks
                           for i in range(len(toks) - 1)}) / max(1, total_toks),
        "mean_length": total_toks / len(hypotheses),
        "repeated_4gram_frac": sum(rep_fracs) / len(rep_fracs) if rep_fracs else 0.0,
    }


if __name__ == "__main__":
    hyps = [
        "the heart size is normal. the lungs are clear. no pleural effusion.",
        "mild cardiomegaly. small left pleural effusion.",
    ]
    refs = [
        "heart size is normal. lungs are clear. no pleural effusion or pneumothorax.",
        "the heart is mildly enlarged. there is a small left pleural effusion.",
    ]
    for k, v in nlg_metrics(hyps, refs, prefer_coco=False).items():
        print(f"{k:10s} {v:.4f}")
    ce = clinical_efficacy(hyps, refs)
    print(f"CE micro P/R/F1: {ce['micro_precision']:.3f} / {ce['micro_recall']:.3f} / "
          f"{ce['micro_f1']:.3f}")
    print("diversity:", {k: round(v, 3) for k, v in diversity_stats(hyps).items()})
