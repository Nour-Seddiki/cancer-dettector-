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
- **Memory:** other apps on this laptop hold about 6.4 GB of the 15.6 GB of RAM
  (`GiMATE_llm` / `GiMATE_ai`), and the harness kills background runs when free memory
  runs low. Phase 2 of report training was killed with 4 loader workers (v8) and again
  with 2 (v9), so run phase 2 with `--num-workers 0`. Phase 1 and the classifier survived
  with 2 workers.

## Current best

`checkpoints/rg_v5a_phase2.pt`, warm-started from `checkpoints/classifier_v2.pt`. The recipe
is in the README "v5" section.

- val: CE micro P / R / F1 0.351 / 0.354 / **0.352**, macro F1 0.235
- test: CE micro P / R / F1 0.323 / 0.323 / **0.323**, macro F1 0.202, BLEU-1 0.246,
  CIDEr 0.310
- The classifier head's own thresholded micro F1 on val is 0.39, so it is the ceiling.

## Queue (highest expected value first)

1. **Combine the two near-misses: head TTA (zoom 0.1) on v10a.** Each fell just short on
   its own (TTA +0.014 on v5a, aux weight 0.5 +0.013), and they act on different parts:
   TTA sharpens the head's probabilities, while the lighter aux loss lets the decoder write
   more findings (recall 0.354 to 0.407). The TTA code has to be re-added (method in the
   v6 note), then `rg_v10a_phase2.pt` is copied, re-tuned with `--tta-zoom 0.1` and
   evaluated on val. Caveat: stacking val near-misses risks multiple comparisons, so if it
   is accepted, the test evaluation is the check that the gain is real.
2. **Seed ensemble of the head.** Average the probabilities of 2-3 phase-2 runs.
3. **Decoder regularisation:** dropout 0.2, or 2 layers.

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
| v9 | 320px input throughout (`classifier_v4` + v5a recipe, 100 image tokens) | 0.305 | - | rejected |
| v10a / v10b | v5a recipe with aux weight 0.5 / 2.0 (was 1.0) | 0.365 / 0.258 | - | rejected, v10a near-miss |

\* Scored before the labeller fix. Test numbers for v2 were re-scored with the fixed labeller.

**v6 (head TTA).** At inference, the classifier head's probabilities were averaged over
the original view plus 5 crops (centre and four corners) of a copy upscaled by the zoom
factor, with no horizontal flip. Thresholds were then re-tuned on the averaged
probabilities, and the decoder's image tokens stayed on the original view. On val, zoom
0.10 moved the head's own micro F1 from 0.391 to 0.411 and the reports from 0.352 to
0.366 (precision 0.351 to 0.382, recall 0.354 to 0.351, macro 0.235 to 0.252). Zoom 0.15
gave 0.363. The +0.014 gain is under the 0.015 bar, so it was rejected and reverted, even
though both zooms agree.

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

**v9 (320px input).** An `image_size` setting was threaded through the transforms,
datasets, loaders, encoder grid, model config and every inference entry point, then
reverted. At 320px the 10x10 grid gives 100 image tokens. `classifier_v4` was trained at
320px with the classifier_v2 recipe unchanged (batch 32; peak VRAM for a batch of 16 was
2.45 GB). It reached a best val mean AUROC of 0.716, against 0.709 at 224px. The v5a recipe
at 320px, warm-started from it, gave a head with thresholded micro F1 0.404 on val (was
0.391; pleural effusion 0.522, lung opacity 0.488). The reports were worse, though: CE
micro P / R / F1 0.343 / 0.275 / 0.305 and macro 0.205, so the decoder passed through about
75% of the head's F1. Pneumothorax and fracture stayed at zero even at 320px. At this data
size, more pixels mainly give the decoder more tokens to attend over, not better findings.

**v10 (aux weight).** This was a command-line change only, so there was no code to revert.
The v5a recipe was re-run with `--aux-weight` 0.5 and 2.0. The heavier weight hurt badly:
CE micro P / R / F1 0.416 / 0.187 / 0.258, macro 0.165. Pulled harder toward the manifest
labels, the head made the decoder cautious. The lighter weight gave the best val result
yet, CE micro P / R / F1 0.332 / 0.407 / 0.365 and macro 0.240. Recall went from 0.354 to
0.407 at a small precision cost, and it wrote 33% distinct reports against v5a's 27%. The
head's own micro F1 was 0.400. At +0.013 it is under the 0.015 bar, so it was rejected,
and `rg_v10a_phase2.pt` is kept for the combination test in queue item 1.
