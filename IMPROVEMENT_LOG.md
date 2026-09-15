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
  runs to P-cores with affinity `0x0FFF`. Use `--num-workers 2` for report-generator
  training: with 4 workers, phase 2 of v8 was killed for low system memory (other apps on
  this laptop hold ~6 GB of the 15.6 GB).

## Current best

`checkpoints/rg_v5a_phase2.pt`, warm-started from `checkpoints/classifier_v2.pt`. The recipe
is in the README "v5" section.

- val: CE micro P / R / F1 0.351 / 0.354 / **0.352**, macro F1 0.235
- test: CE micro P / R / F1 0.323 / 0.323 / **0.323**, macro F1 0.202, BLEU-1 0.246,
  CIDEr 0.310
- The classifier head's own thresholded micro F1 on val is 0.39, so it is the ceiling.

## Queue (highest expected value first)

1. **Higher input resolution.** For example 320px, which gives a 10x10 grid of 100 image
   tokens. This needs `GRID` and `NUM_REGIONS` to follow the input size.
2. **Auxiliary-loss weight sweep** (0.5, 2.0) on the v5a recipe.
3. **Seed ensemble of the head.** Average the probabilities of 2-3 phase-2 runs.
4. **Decoder regularisation:** dropout 0.2, or 2 layers.
5. **Retest head TTA on the next accepted model.** Zoom 0.1 was a near-miss on v5a (see
   History); the method is described in the v6 note and is inference only.

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
| v6a / v6b | head TTA at inference, zoom 0.10 / 0.15 | 0.366 / 0.363 | - | rejected, near-miss |
| v7 | warm-start from `classifier_v3` (trained on text labels), v5a recipe | 0.301 | - | rejected |
| v8 | `classifier_v3` + text-label tokens + text-label aux BCE | 0.346 | - | rejected |

\* Scored before the labeller fix. Test numbers for v2 were re-scored with the fixed labeller.

**v6 (head TTA).** At inference, the classifier head's probabilities were averaged over
the original view plus 5 crops (centre and four corners) of a copy upscaled by the zoom
factor, with no horizontal flip. Thresholds were then re-tuned on the averaged
probabilities, and the decoder's image tokens stayed on the original view. On val, zoom
0.10 moved the head's own micro F1 from 0.391 to 0.411 and the reports from 0.352 to
0.366 (precision 0.351 to 0.382, recall 0.354 to 0.351, macro 0.235 to 0.252). Zoom 0.15
gave 0.363. The +0.014 gain is under the 0.015 bar, so it was rejected and reverted, even
though both zooms agree. It is queued to retest on top of the next accepted model.

**v7 (text-label classifier).** `classifier_v3` was trained exactly like `classifier_v2`
(15 epochs), but on the rule-based labeller's reading of each report instead of the
manifest's MeSH-first labels, via a `--label-source text` option in `train_classifier.py`
(reverted, while `classifier_v3.pt` is kept). Its best val mean AUROC was 0.714, against
text labels. Warm-started from it, the v5a recipe gave a better head, with thresholded
micro F1 on val of 0.422 (was 0.391; cardiomegaly 0.575, lung opacity 0.473, atelectasis
0.480). The reports were worse, though: CE micro P / R / F1 0.416 / 0.236 / 0.301 and macro
0.188. The decoder passed through 71% of the head's F1, where v5a passed about 90%. The
likely cause was that the decoder was taught with manifest-label tokens while the head
predicted text labels, and the aux BCE on manifest labels pulled the head back.

**v8 (text labels end to end).** It tested that explanation: the same `classifier_v3`
warm start, with the condition tokens during training (`--teacher-source text`) and the
aux BCE (an `--aux-source text` option, reverted) both on text labels. Matching the
labels did restore the pass-through, with reports at 85% of the head's F1 (0.346 of 0.408),
but the head itself ended weaker than v7's, at 0.408. The reports came out at CE micro P /
R / F1 0.320 / 0.377 / 0.346 and macro 0.200. That is higher recall than v5a but lower
precision, and below it overall. With v5b, v7 and v8 all short of the manifest-label v5a,
text-derived labels are not a lever at this data size, and that line is closed.
