# Improvement log

Working notes for the model-improvement loop. Each round tries one idea from the queue.
This file holds the rules, the current best model, the queue and every result so far.

## Rules

- **Select on val, never on test.** The headline metric is clinical-efficacy (CE) micro F1
  over the 13 abnormal conditions (rule-based labeller), with greedy decoding,
  `no_repeat_ngram=3`, and condition thresholds tuned on val by `tune_thresholds.py`.
- **Accept** an idea if val CE micro F1 beats the current best by at least 0.015 without
  macro F1 dropping. Smaller gaps are within val noise (361 studies). Only an accepted
  model gets a test evaluation, and it is recorded below.
- **Accepted:** update the README results, then commit and push without co-author trailers.
  **Rejected:** revert the code change, but keep the log entry.
- New checkpoints get new names (`rg_vN_*`, `classifier_vN.pt`). Never overwrite an old one.
- Run Python with the signed interpreter and the venv on `PYTHONPATH`
  (`C:\Users\seddi\AppData\Local\Programs\Python\Python314\python.exe`). Pin long background
  runs to P-cores with affinity `0x0FFF`.

## Current best

`checkpoints/rg_v5a_phase2.pt`, warm-started from `checkpoints/classifier_v2.pt`. The recipe
is in the README "v5" section.

- val: CE micro P / R / F1 0.351 / 0.354 / **0.352**, macro F1 0.235
- test: CE micro P / R / F1 0.323 / 0.323 / **0.323**, macro F1 0.202, BLEU-1 0.246,
  CIDEr 0.310
- The classifier head's own thresholded micro F1 on val is 0.39, so it is the ceiling.

## Queue (highest expected value first)

1. **Test-time augmentation for the classifier head.** Average its probabilities over a few
   crops or scales (no horizontal flip, because anatomy is left-right dependent), then
   re-tune the thresholds. Inference only, no retraining.
2. **Stronger stage (a) classifier.** Retrain `classifier_v3` on the text-derived labels
   (what CE measures) and/or for longer, then warm-start v6 from it.
3. **Higher input resolution.** For example 320px, which gives a 10x10 grid of 100 image
   tokens. This needs `GRID` and `NUM_REGIONS` to follow the input size.
4. **Auxiliary-loss weight sweep** (0.5, 2.0) on the v5a recipe.
5. **Seed ensemble of the head.** Average the probabilities of 2-3 phase-2 runs.
6. **Decoder regularisation:** dropout 0.2, or 2 layers.

## History

| version | change | val micro F1 | test micro F1 | verdict |
|---|---|---|---|---|
| v2 | view fix + retrained classifier, beam 3 | 0.062* | 0.065 | baseline |
| v2 | greedy decoding | 0.201* | 0.206 | accepted |
| v3 | soft condition tokens + aux BCE | 0.212* | - | superseded |
| v4 | ground-truth tokens for 50% of studies, binary tokens | 0.314* | - | superseded |
| v5a | ground-truth tokens for 100% of studies | 0.340* | 0.317* | accepted |
| v5b | teacher labels read off the report text | 0.301* | - | rejected |
| v5a | fixed cardiomegaly labeller + `no_repeat_ngram=3` | 0.352 | 0.323 | accepted |

\* Scored before the labeller fix. Test numbers for v2 were re-scored with the fixed labeller.
