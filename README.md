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

Not yet run: training on the real dataset.

## References

- IU X-Ray / Open-i — <https://openi.nlm.nih.gov/>
- MIMIC-CXR (credentialed) — <https://physionet.org/content/mimic-cxr/2.0.0/>
- R2Gen, the reference architecture and CE implementation — <https://github.com/zhjohnchan/R2Gen>
- CheXbert — <https://github.com/stanfordmlgroup/CheXbert>
