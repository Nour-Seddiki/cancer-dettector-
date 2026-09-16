# Chest X-Ray Report Generation (CNN Encoder → Transformer Decoder)

Generates a free-text radiology report from a chest X-ray — not just a class label.
DenseNet121 (ImageNet-pretrained) encodes the image into 49 spatial tokens; a from-scratch
Transformer decoder cross-attends over them and writes the report.

The decoder blocks are carried over from the [EN→FR translation project](../transformer)
almost unchanged — that is the point of the design. A decoder that cross-attends over "a
sequence of context vectors" does not care whether that sequence came from a text encoder
or from a CNN's flattened feature grid. `VisualBridge` is the entire adapter.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full rationale and week-by-week plan.

## Architecture

```
X-ray (1×224×224, grayscale)
   │  replicate to 3 channels, ImageNet normalise
   ▼
DenseNet121 .features                      → (B, 1024, 7, 7)
   │  F.relu, flatten the spatial grid      (no global average pool —
   ▼                                         pooling throws away *where*)
49 spatial tokens                          → (B, 49, 1024)
   │  VisualBridge: Linear(1024→512) + LayerNorm + learned grid position
   ▼
context                                    → (B, 49, 512)
   │
   ├──────────► DecoderBlock × 3  ◄── report tokens (causal self-attention)
   │              · causal self-attention
   │              · cross-attention over the 49 image tokens
   │              · feed-forward
   ▼
LayerNorm → lm_head (tied to token embedding) → (B, T, vocab≈1.5k)
```

~21M parameters total (7M DenseNet121 + ~14M decoder).

Why the decoder is smaller than the translation model's (3×512 vs 6×384, and a ~1.5k vocab
instead of ~100k): IU X-Ray has roughly 3k training studies. A bigger stack overfits long
before it converges. The small vocab also makes the `B*T*vocab_size` logits blowup from the
translation project mostly disappear.

## Setup

```bash
pip install -r requirements.txt
python prepare_data.py        # downloads IU X-Ray (~2 GB) and builds the manifest
python eda.py --plots         # sanity-check the data before training anything
```

The dataset lands in `data/` by default. That directory is gitignored, but on this machine
the project lives inside OneDrive, which will happily try to sync a 1.4 GB tarball and 7,470
PNGs. Point it elsewhere:

```bash
# PowerShell, persists for the user
setx CXR_DATA_DIR "C:\ml-data\iu-xray"
```

`prepare_data.py` pulls the PNG images and ecgen-radiology XML reports from
[Open-i](https://openi.nlm.nih.gov/) (public, no credentialing), extracts FINDINGS +
IMPRESSION as the generation target, maps the shipped MeSH terms onto the 14 CheXpert
conditions, and writes `data/processed/manifest.csv`.

**Splits are assigned by study, not by image** (`stable_split`, an MD5 hash of the study
uid). A study's frontal and lateral views are near-duplicates; splitting by image would put
one in train and the other in test and silently inflate every number.

## Training

Three stages, in this order. Each one is independently checkable, which is the whole reason
for the ordering — if stage (a) is at chance, the bug is in the image pipeline, and no
amount of decoder debugging will find it.

### (a) CNN classifier baseline — local GPU

```bash
python train_classifier.py --epochs 15 --batch-size 32
```

Fine-tunes DenseNet121 for 14-way multi-label classification and reports per-class AUROC
(CheXNet's own metric). Conditions with fewer than 10 positives are excluded from the mean —
their AUROC is noise. Starts with the backbone frozen for one epoch so the randomly
initialised head does not push garbage gradients into pretrained features.

### (c) Report generator — debug small, then scale

```bash
# 1. does the architecture learn at all? loss should approach 0 in a few hundred steps
python training.py --overfit 64 --epochs 200 --batch-size 8 --num-workers 0

# 2. phase 1: frozen CNN, full dataset
python training.py --epochs 30 --cnn-checkpoint checkpoints/classifier.pt

# 3. phase 2: unfreeze denseblock4 + norm5 at a much lower LR
python training.py --epochs 15 --resume checkpoints/report_generator.pt \
    --unfreeze-cnn --lr 1e-4 --cnn-lr 1e-5
```

The overfit run is the highest-value debugging step in the project. If 64 examples will not
memorise, something is wrong with masking, the label shift, or the shapes — and finding that
out costs two minutes instead of a whole training run.

Phase ordering matters: a frozen CNN gives the decoder stable features to learn the language
of radiology against, and it cannot overfit the backbone to ~3k studies. Only unfreeze once
the generated text is coherent, and then at a 10–30× smaller LR than the decoder
(`model.param_groups` sets up the discriminative LRs).

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/report_generator.pt --split test --beam-size 3
python generate.py --image data/raw/images/CXR1_1_IM-0001-3001.png --show-labels
```

Two tiers of metric, and the distinction matters:

**NLG metrics** (BLEU-1..4, ROUGE-L, CIDEr) are the fast iteration signal. They are also
misleading in isolation: most chest X-rays are normal, so a model that emits "no acute
cardiopulmonary abnormality" for every study scores a respectable BLEU while being clinically
useless. `eda.py` prints that floor — judge against it, not against zero.

**Clinical efficacy** is the headline number. Label both generated and reference reports with
the 14 conditions, then score precision/recall/F1. "No Finding" is excluded from the micro
average for exactly the reason above.

`evaluate.py` also prints diversity statistics (distinct-1/2, repeated-4-gram fraction,
fraction of unique reports) to catch the two classic captioning failure modes — repetition
loops and mode collapse onto one template — and the five worst examples by ROUGE-L as a
starting point for error analysis.

### Upgrading the labeller

`labels.py` is a dependency-free, negation-aware rule-based labeller. The literature standard
is [CheXbert](https://github.com/stanfordmlgroup/CheXbert), which needs a downloaded BERT
checkpoint. To swap it in, implement a callable with the same signature
(`str -> list[int]` of length 14, in `labels.CONDITIONS` order) and pass it:

```python
clinical_efficacy(hypotheses, references, labeller=chexbert_labeller)
```

Nothing else changes. Report numbers as "rule-based CE" until you do — they are not directly
comparable to published CheXbert CE figures.

## Files

| file | what it does |
|---|---|
| `prepare_data.py` | download IU X-Ray, parse reports, build the manifest, split by study |
| `eda.py` | week-1 EDA: lengths, vocab, prevalence, boilerplate floor |
| `labels.py` | the 14 CheXpert conditions; MeSH and rule-based text labellers |
| `dataloader.py` | image preprocessing, word-level tokenizer, paired dataset, collation |
| `cnn_model.py` | DenseNet121 classifier (stage a) + 49-token feature extractor (stage b) |
| `model.py` | `VisualBridge` + decoder (reused from `transformer/model.py`) + beam search |
| `train_classifier.py` | stage (a) training loop, per-class AUROC |
| `training.py` | stage (c) training loop, two-phase freeze/unfreeze |
| `evaluate.py` | generate on a split, score NLG + clinical efficacy, error analysis |
| `metrics.py` | BLEU / ROUGE-L / CIDEr-D / clinical efficacy implementations |
| `utils.py` | checkpointing, LR schedule, GPU/OS workarounds |

`metrics.py` and `eda.py` are additions to the file list in the plan's §6 — week 6 needs a
metrics implementation, and the week-1 EDA deserved to be reproducible rather than a notebook.

## Notes on this machine

Carried over from the translation project, and baked into `utils.py`:

- **`cap_gpu_memory`** — on Windows/WDDM the driver silently spills past physical VRAM into
  shared system RAM instead of raising OOM, turning a run into a PCIe-bound crawl with no
  error to point at. Capping the allocator at 80% makes it fail fast so you lower the batch
  size instead. `--gpu-fraction` tunes it.
- **`keep_awake`** — Windows Modern Standby suspends a run after ~30 idle minutes even with
  the GPU saturated, and often kills it on wake. `SetThreadExecutionState` holds it off.
- **`save_checkpoint`** — OneDrive transiently locks large files while syncing; saves retry
  rather than crashing at the end of a long run.
- The RTX 5050 is power-capped to ~35 W on battery. Plug in for any real run.
- Launching a run in the background can land it on E-cores and run ~5× slower; pin to
  P-cores (`start /affinity 0x0FFF`) for long jobs.

## Status

Implemented and verified end-to-end on synthetic data: manifest build, tokenizer round-trip,
teacher-forcing shift, frozen/unfrozen gradient flow, greedy + beam decoding, checkpoint
round-trip, and all metrics (BLEU/ROUGE/CIDEr = max on identical text, ≈0 on mismatched).

Trained on IU X-Ray (3,101 train / 374 val / 351 test studies, frontal only):
phase 1 (CNN frozen) reached val loss 2.5053, phase 2 (denseblock4+norm5 unfrozen,
lr 1e-4 / cnn-lr 1e-5) improved it to 2.4676. Both early-stopped; ~4.5 min total on
the RTX 5050.

**The trained model has mode-collapsed and is not clinically useful.** It emits 5
distinct reports across all 351 test studies (`unique_report_frac` 0.014 vs 0.900 for
references) and predicts "No Finding" for every one of them, so clinical efficacy is
0.000 on all 13 abnormal conditions. NLG metrics beat the constant-boilerplate floor
(BLEU-1 0.280 vs 0.140, ROUGE-L 0.240 vs 0.200) but fall *below* it on CIDEr
(0.237 vs 0.266) - which is the giveaway that the gain is fluency, not grounding.

This is the loss-optimal degenerate solution for a 3,101-study set that is 48%
"No Finding": plain token cross-entropy is minimised by always writing the normal
template. Next steps are abnormality-weighted sampling, the auxiliary BCE classifier
loss from the plan's Section 4, and confirming the decoder is using the image at all
rather than having learned a pure language prior.

### v2: fixed view selection + a working stage (a) classifier

Running the model on held-out X-rays (`python demo.py`) exposed two upstream bugs:

- **15% of the "frontal-only" manifest was not frontal.** The old rule took each study's
  lowest-sorting image id as its frontal, assuming ids follow acquisition order; they
  don't. 589 of 3,826 rows were laterals (439 studies had the wrong image picked, 150 had
  no frontal at all). `prepare_data.py` now reads the view from the pixels: a logistic
  regression on 32×32 thumbnails, self-trained on the within-study mirror-symmetry gap
  (grouped CV 0.99). Lateral-only studies are dropped. Splits are unchanged for every
  remaining study: 2,983 train / 361 val / 332 test.
- **`classifier.pt` was at chance** (mean AUROC 0.504). It was the epoch-0 checkpoint of a
  run that died right after its frozen-backbone warm-up epoch, and the report generator
  had been warm-started from it. The training code itself is fine: the full 15-epoch run
  (`classifier_v2.pt`) reaches mean val AUROC 0.709, with Cardiomegaly 0.90, Pleural
  Effusion 0.93 and Atelectasis 0.80.

The report generator was retrained on the fixed data with the w3 recipe (abnormal studies
oversampled ×3, frozen CNN, then denseblock4+norm5), warm-started from `classifier_v2.pt`,
giving `rg_v2_phase2.pt` (val loss 2.5698). Scores below are on the test split (332
studies, beam 3, rule-based CE labeller), with the previous model evaluated on the same
split:

| | `rg_w3_phase2` | `rg_v2_phase2` |
|---|---|---|
| distinct reports | 12 (3.6%) | 31 (9.3%) |
| CE micro P / R / F1 | 0.200 / 0.004 / 0.007 | 0.636 / 0.026 / 0.051 |
| Cardiomegaly F1 | 0.000 | 0.300 |
| BLEU-1 / BLEU-4 | 0.155 / 0.051 | 0.212 / 0.060 |
| ROUGE-L / CIDEr | 0.255 / 0.356 | 0.237 / 0.258 |

v2 is better grounded and less collapsed, but it is still not useful: it reports only 2.6%
of abnormal findings. The classifier on the same features now detects cardiomegaly and
effusion well, yet the decoder rarely writes them. The bottleneck has therefore moved from
the image features to the decoder's normal-template prior. Next steps are the auxiliary BCE
loss from the plan and conditioning the decoder on the classifier's predictions.

### v5: condition tokens from the classifier head

The decoder now cross-attends over 63 tokens: the 49 image regions plus 14 condition
tokens (`--label-tokens`, `model.LabelBridge`). Each condition token interpolates between
learned "present" and "absent" embeddings according to the CNN classifier head's output,
and the head keeps training through the plan's auxiliary BCE loss (`--aux-weight`).

The steps that mattered, measured on val (clinical-efficacy micro F1 over the 13 abnormal
conditions):

- **Greedy decoding instead of beam 3:** 0.062 → 0.201 on the v2 model, with no
  retraining. Beam search converges on the highest-likelihood report, which here is the
  normal template. Greedy is now the default in `evaluate.py`, `generate.py` and `demo.py`.
- **Soft tokens from the head's own predictions (v3): 0.212.** Diagnostic: giving v3 the
  ground-truth labels as tokens changes nothing (0.196). Trained only on its own noisy
  predictions, it had learned to ignore the tokens.
- **Ground-truth tokens during training (`--label-teacher-prob`):** with half of the
  training studies on ground truth (v4), feeding the true labels at eval reaches 0.400, so
  the decoder now follows the tokens. With all of them on ground truth (v5a), plus binary
  tokens at inference, it reaches **0.340**.
- **Binary tokens at thresholds tuned on val (`tune_thresholds.py`), chosen jointly to
  maximise micro F1.** Tuning each condition's F1 on its own set Pneumonia to a threshold
  that flagged 34% of studies at 4% precision, which put the finding into every one of
  those reports. "No Finding" is derived from the abnormal tokens (it means "none of them"
  in every ground-truth label); on v4 that scored 0.314 against 0.289 for thresholding it.
- Taking the teacher labels from the report text instead of the manifest (v5b) did not
  help (0.301).

```bash
python training.py --epochs 30 --abnormal-weight 3 --label-tokens --aux-weight 1 \
    --label-teacher-prob 1 --cnn-checkpoint checkpoints/classifier_v2.pt \
    --out checkpoints/rg_v5a_phase1.pt
python training.py --epochs 15 --abnormal-weight 3 --label-tokens --aux-weight 1 \
    --label-teacher-prob 1 --resume checkpoints/rg_v5a_phase1.pt --unfreeze-cnn \
    --lr 1e-4 --cnn-lr 1e-5 --out checkpoints/rg_v5a_phase2.pt
python tune_thresholds.py --checkpoint checkpoints/rg_v5a_phase2.pt   # val
python evaluate.py --checkpoint checkpoints/rg_v5a_phase2.pt --split test
```

Two fixes came out of testing examples by hand:

- **Labeller:** "the heart is mildly enlarged" and "heart is large" did not count as
  cardiomegaly, because the pattern allowed no degree word between the verb and
  "enlarged". Against the shipped MeSH labels, cardiomegaly recall goes from 0.857 to
  0.939 at unchanged precision. Every number below uses the fixed labeller.
- **Repetition:** reports sometimes looped ("left lung is clear. left lung is clear.").
  Decoding now forbids repeating any 3-gram (`no_repeat_ngram=3`, the winner of a val
  sweep over off/3/4/5/6). Repeated 4-grams drop from 1.4% to 0, and val CE micro F1
  moves from 0.338 to 0.352.

Test split (332 studies, rule-based CE labeller, thresholds tuned on val, test scored once
per model):

| | v2, beam 3 (previous) | v2, greedy | **v5a** |
|---|---|---|---|
| CE micro P / R / F1 | 0.692 / 0.034 / 0.065 | 0.479 / 0.132 / 0.206 | 0.323 / 0.323 / **0.323** |
| CE macro F1 | 0.035 | 0.108 | 0.202 |
| distinct reports | 9.3% | 17.2% | 25.9% |
| BLEU-1 / BLEU-4 | 0.212 / 0.060 | 0.191 / 0.061 | 0.246 / 0.071 |
| ROUGE-L / CIDEr | 0.237 / 0.258 | 0.261 / 0.359 | 0.255 / 0.310 |

Recall on abnormal findings went from 3.4% to 32%, and v5a has no repeated 4-grams. By
condition, v5a scores F1 0.46 on cardiomegaly, 0.44 on lung opacity, 0.41 on atelectasis
and 0.34 on pleural effusion. The classifier head is now the ceiling: its own thresholded
micro F1 on val is 0.39, and the reports reach 0.35. The model still misses almost all
lung lesions (F1 0.06), pneumothorax, pneumonia and fractures, and it writes some findings
into normal studies. The next levers are a stronger classifier (higher input resolution,
test-time augmentation) and scoring with CheXbert instead of the rule-based labeller.
`IMPROVEMENT_LOG.md` tracks the experiments.

### v12: one shared condition threshold

The 14 condition tokens are switched on by thresholding the classifier head's probabilities,
and until v12 those thresholds were fitted one per condition, by coordinate ascent on micro
F1 over the 361 val studies. That is 13 free parameters fitted on a few hundred examples,
several conditions carrying fewer than 25 positives — and it was overfitting badly enough to
both hide real gains and manufacture fake ones.

Measured directly: moving a val-fitted threshold vector to the test split costs about **0.06
head micro F1** (0.414 → 0.353), more than any modelling change in this project's history.
The per-condition optima simply move between splits — pleural effusion 0.575 → 0.350,
atelectasis 0.700 → 0.475, lung lesion 0.550 → 0.925.

Fitting a *single* threshold shared by all 13 conditions — one free parameter — was chosen by
fitting on one random half of val and scoring on the other, 30 times, where it beat the
per-condition fit by +0.020 micro F1. It could not be chosen on full-val CE, because that is
the number the thresholds are fitted against.

End to end on the test split, with the model weights completely unchanged:

| | v5a (13 thresholds) | **v12a (1 threshold)** |
|---|---|---|
| CE micro P / R / F1 | 0.323 / 0.323 / 0.323 | 0.358 / 0.312 / **0.333** |
| CE macro F1 | 0.202 | 0.217 |
| BLEU-4 / CIDEr | 0.071 / 0.310 | 0.071 / 0.339 |

```bash
python tune_thresholds.py --checkpoint checkpoints/rg_v12a_global.pt --rule global
python evaluate.py --checkpoint checkpoints/rg_v12a_global.pt --split test
```

The val score goes *down* (0.352 → 0.317) while the test score goes up, which is the whole
point: the val figure was partly in-sample. v12a's test score is above its val score.

The same threshold change on the aux-weight-0.5 + head-TTA weights (`v12b`, the current best
by val) also reaches 0.333 test micro F1, up from 0.319. Head TTA and the lighter auxiliary
loss each genuinely improve the classifier head, and TTA's head gain does transfer to test —
but neither survives the decoder as a report-level gain. The threshold rule does.

## References

- IU X-Ray / Open-i — <https://openi.nlm.nih.gov/>
- MIMIC-CXR (credentialed) — <https://physionet.org/content/mimic-cxr/2.0.0/>
- R2Gen, the reference architecture and CE implementation — <https://github.com/zhjohnchan/R2Gen>
- CheXbert — <https://github.com/stanfordmlgroup/CheXbert>
