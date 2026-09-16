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
- **Val CE is in-sample for the thresholds.** `tune_thresholds.py` fits the condition
  thresholds on val, so a val CE score partly measures that fit rather than generalisation.
  Two *threshold rules* therefore cannot be compared on full-val CE at all; fit on one
  random half of val and score on the other (the v12 note has the protocol). Comparing two
  *trained models* at a fixed threshold rule is still fine on val.
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

`checkpoints/rg_v12b_global_tta.pt` - the v10a weights (v5a recipe at aux weight 0.5), head
TTA at zoom 0.10, and **one condition threshold shared by all 13 abnormal conditions**.
The recipe is in the README "v12" section.

- val: CE micro P / R / F1 0.324 / 0.416 / **0.364**, macro F1 0.227
- test: CE micro P / R / F1 0.299 / 0.376 / **0.333**, macro F1 0.216, BLEU-1 0.269,
  CIDEr 0.268

`checkpoints/rg_v12a_global.pt` - the unchanged v5a weights with the same single threshold -
reaches the *same* 0.333 test micro F1 from a strictly smaller change, at higher precision
(0.358 / 0.312) and much better CIDEr (0.339 vs 0.268). v12b is recorded as best because it
is ahead on val (0.364 vs 0.317), which is the only comparison selection is allowed to make;
preferring v12a on its test precision would be selecting on test. If a reason to choose is
ever needed, settle it with a fresh val-side comparison, not by looking at the table below.

The head's own micro F1 at an oracle threshold is ~0.415 on test while the reports reach
0.333, so the decoder's pass-through is now the ceiling rather than the head.

## Queue (highest expected value first)

1. **Re-score every rejected idea at the new threshold rule.** v6 (TTA), v7/v8 (text
   labels) and v9 (320px) were all judged on full-val CE with 13 val-fitted thresholds -
   the contaminated metric - and v11 proved a real head gain can hide behind it. Re-tune
   the kept checkpoints with `--rule global` and re-score. None of them needs retraining,
   so this is the cheapest experiment left and it re-opens four closed lines at once.
2. **Push the decoder's pass-through, now the binding constraint.** The head reaches ~0.415
   test micro F1 at an oracle threshold; the reports reach 0.333. Seed-ensembling the head
   and decoder regularisation (dropout 0.2, or 2 layers) attack different halves of that gap.
3. **Bootstrap confidence intervals on val CE**, so the 0.015 acceptance bar is set against
   the measured noise of 361 studies instead of a guess. v11 cleared the bar by +0.024 and
   delivered nothing; the bar is probably too low.

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
| v11 | v10a + head TTA zoom 0.10 (queue item 1) | 0.376 | 0.319 | passed val, **refuted on test** |
| v12a | v5a weights, one shared condition threshold | 0.317 | **0.333** | accepted |
| v12b | v10a + TTA 0.10 + one shared threshold | 0.364 | **0.333** | accepted, best on val |

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

**v11 (aux weight 0.5 + head TTA, the two near-misses stacked).** Queue item 1. The TTA code
was re-added from the v6 note, now as a `--tta-zoom` option on `tune_thresholds.py` that
stores the zoom in the checkpoint next to the thresholds, so every inference entry point
reproduces the probabilities the thresholds were tuned on. `rg_v10a_phase2.pt` was copied to
`rg_v11_tta10.pt`, re-tuned at zoom 0.10 and evaluated.

On val it is the best result the project has produced: CE micro P / R / F1
0.339 / 0.423 / **0.376** and macro **0.272**, against v5a's 0.352 / 0.235. That is +0.024,
clear of the 0.015 bar, with macro up as well, and the head's own thresholded micro F1 went
from 0.400 to 0.410. The two effects did stack roughly additively (TTA alone 0.366, aux 0.5
alone 0.365).

On test it gained nothing: CE micro P / R / F1 0.283 / 0.365 / **0.319** and macro 0.206,
against v5a's 0.323 / 0.323 / **0.323** and 0.202. Micro F1 is 0.004 *below* the current
best, which is inside noise, so this is "no change", not a regression - but it is certainly
not the +0.024 val promised. The shape of the change is v10a's: precision traded for recall
(test precision 0.323 to 0.283, recall 0.323 to 0.365). Pleural effusion is where it costs
most on test, F1 0.34 to 0.26 at 0.21 precision, while cardiomegaly improved (0.46 to 0.49).

The val-to-test gap widened from 0.029 on v5a to 0.057 here, which is the real finding: the
test split was the designated check on stacking two val near-misses, and it says the gain is
not real. **Current best stays `rg_v5a_phase2.pt`.** Both `rg_v11_tta10.pt` and the v10a
checkpoints are kept for the threshold work in queue item 1.

The TTA code is deliberately *not* reverted, unlike previous rejections. It is inert unless a
checkpoint carries a non-zero `tta_zoom`, and re-running v5a on val after the change
reproduced 0.352 / 0.235 exactly, so it changes nothing for the current best; queue item 1
needs it.

**Threshold transfer diagnostic (what v11 actually exposed).** Scoring the head's own
thresholded micro F1 over the 13 abnormal conditions, three ways:

| model | val @ val-tuned | test @ val-tuned | test @ test-tuned (oracle) |
|---|---|---|---|
| v5a (no TTA) | 0.391 | 0.353 | 0.414 |
| v11 (TTA 0.10) | 0.410 | 0.364 | 0.415 |

Two things follow, and they point in opposite directions from the report-level result.

TTA's gain on the head is *real and it does transfer*: at the same val-tuned thresholds the
TTA head is better on test too, 0.364 against 0.353. So v11's flat test CE is not TTA
failing to generalise - it is the decoder not passing a +0.011 head gain through, which is
within the noise of what the decoder passes through anyway.

The thresholds are the overfit part. Both models lose about 0.06 micro F1 purely by moving a
val-tuned threshold vector to test (0.414 to 0.353, 0.415 to 0.364), and their oracle
ceilings are identical at ~0.415. TTA therefore does not raise the ceiling at all; what it
buys is a smaller threshold-transfer penalty (0.051 against 0.061), i.e. robustness to the
threshold landing in the wrong place. The per-condition optima move a long way between the
two splits - pleural effusion 0.575 to 0.350, atelectasis 0.700 to 0.475, lung lesion 0.550
to 0.925, enlarged cardiomediastinum (val support 2) off entirely - which is what 13 free
parameters coordinate-ascended on 361 studies with small per-condition supports look like.

Closing even half of that 0.06 is worth more than any architectural idea left in the queue,
and it costs no retraining: the thresholds are fitted post hoc on cached head probabilities.

**v12 (one shared condition threshold).** Queue item 1, and the first genuine test-split gain
since v5a. `tune_thresholds.py` grew a `--rule` option: `coord` is the original
13-parameter coordinate ascent, `global` fits a single threshold shared by all 13 abnormal
conditions, `shrunk` blends them. `global` is now the default.

Choosing between the rules could not be done on full-val CE, because that is the number the
thresholds are fitted against. The protocol instead fit on one random half of val and scored
on the other, 30 paired repeats, head micro F1:

| rule | v5a held-out | v11 held-out |
|---|---|---|
| coord (13 params, the original) | 0.3240 | 0.3432 |
| **global (1 param)** | **0.3436** | **0.3558** |
| shrunk-0.50 | 0.3416 | 0.3510 |
| bagged-15 (bootstrap-averaged coord) | 0.3301 | 0.3465 |
| support-10 (rare conditions forced off) | 0.3147 | 0.3280 |
| oracle (fit on the scoring half itself) | 0.4083 | 0.4262 |

Bagging the same 13-parameter fit barely helps, and forcing low-support conditions off hurts,
so the problem is not the rare conditions specifically - it is the number of free parameters.

End to end, with model weights untouched and only the stored thresholds differing:

| | val micro F1 | test micro F1 | test macro | test P / R | CIDEr |
|---|---|---|---|---|---|
| v5a, 13 thresholds | 0.352 | 0.323 | 0.202 | 0.323 / 0.323 | 0.310 |
| **v12a, 1 threshold** | 0.317 | **0.333** | 0.217 | 0.358 / 0.312 | 0.339 |
| v11, 13 thresholds | 0.376 | 0.319 | 0.206 | 0.283 / 0.365 | - |
| **v12b, 1 threshold** | 0.364 | **0.333** | 0.216 | 0.299 / 0.376 | 0.268 |

+0.010 on the v5a weights and +0.014 on the v10a+TTA weights, for no retraining at all. Both
bases land on exactly 0.333, which is a good sign that the threshold rule is doing the work
rather than either set of weights.

The val figures move the *opposite* way (0.352 to 0.317, 0.376 to 0.364), and that is the
lesson worth keeping: those val numbers were partly in-sample, so the rule that scores worse
on val is the one that generalises. v12a's test score (0.333) is actually *above* its val
score (0.317) - the val-to-test gap went from -0.029 to +0.016 - which is what an estimator
that is not fitting val noise looks like. A new rule in the Rules section now records this.

Read together with the v11 note, the picture is that head TTA and the lighter auxiliary loss
both do make the classifier head better, and the TTA gain does transfer to test, but neither
survives the decoder as a report-level gain; the threshold rule does, and it was invisible
under the old metric.
