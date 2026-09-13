# Chest X-Ray Report Generation (CNN Encoder → Transformer Decoder)

Portfolio project: DenseNet121 (ImageNet-pretrained, fine-tuned) encoder feeding a from-scratch
Transformer decoder that generates a free-text radiology report — not just a class label.

## 0. Modality choice: chest X-ray, not skin/mammography/histopathology

Sticking with chest X-ray is the right call for a *report-generation* project specifically:

- **IU X-Ray** (Indiana University Chest X-Ray Collection, via [Open-i](https://openi.nlm.nih.gov/)) — ~7,470 images / 3,955 studies, each with a real free-text radiology report (Findings/Impression). **Public, no credentialing** — good for week 1 prototyping.
- **MIMIC-CXR** ([PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)) — 377,110 images / 227,835 studies with free-text reports. Much larger, better for the transformer stage, but **requires PhysioNet credentialing**: a credentialed account + "CITI Data or Specimens Only Research" training + a signed Data Use Agreement. **Approval can take days — apply in Week 1**, don't block on it.
- Skin lesion (ISIC) and most histopathology/mammography sets are **classification-labeled only** (a diagnosis category, not a written report) — they don't support "generate a diagnostic description" without collecting/synthesizing report text yourself, which is a much bigger undertaking. Chest X-ray is the only option of the four with mature paired image+report datasets and established benchmarks/tooling (below).

## 1. Week-by-week build plan

### Week 1 — Setup + data
- Apply for MIMIC-CXR PhysioNet access now (credentialing lag). Meanwhile download **IU X-Ray**.
- Write `dataloader.py`: DICOM/PNG loading, train/val/test split (split **by patient/study**, not by image, to avoid leakage between an image's two views).
- EDA: report length distribution, vocab frequency, frontal vs. lateral view counts, class imbalance (most CXRs are "normal").
- Scaffold the repo (see §6) mirroring your `transformer/` project's layout.

### Week 2 — Stage (a): CNN classifier baseline
- Fine-tune DenseNet121 for **multi-label classification** of the standard 14 CheXpert-style findings (Atelectasis, Cardiomegaly, Edema, etc.), using labels either shipped with the dataset or extracted from the reports via the CheXpert/CheXbert labeler (see §4).
- This is a self-contained, evaluable milestone (report per-class AUROC — the CheXNet paper's own metric) *before* you add decoder complexity, so you know the CNN side works in isolation.
- Runs entirely on the **local RTX 5050**.

### Week 3 — Stage (b): feature-extraction bridge
- Freeze the Week 2 DenseNet121, extract the `features` (pre-pool) spatial map (§3), sanity-check shapes and visualize activations/Grad-CAM against known findings.
- Build the text pipeline: tokenizer/vocab from the training reports (word-level, frequency-cutoff — see §3), `(image, report)` paired dataset + collation.
- Still local — this stage is data engineering + inference only, no heavy training.

### Weeks 4–5 — Stage (c): Transformer decoder
- Implement a decoder that cross-attends over the 49 CNN feature tokens — reuse your `transformer/model.py` `DecoderBlock`/`CrossAttention`/`MultiHeadAttention` almost as-is (§3 and §6 explain the swap).
- Debug on a **tiny overfit subset** (50–200 pairs, few hundred steps) locally on the RTX 5050 until it memorizes — this is the fastest way to catch bugs (wrong masking, label/pad handling, shape mismatches) before spending real compute.
- Move full training runs (whole IU X-Ray, and MIMIC-CXR once credentialed) to **Colab T4** (§5).
- Start with the CNN frozen; once the decoder produces coherent text, unfreeze DenseNet121's last dense block (`denseblock4` + `norm5`) with a lower LR for a joint fine-tune pass.

### Week 6 — Evaluation
- Implement BLEU-1..4 / ROUGE-L / CIDEr (`pycocoevalcap`, same library the reference implementation below uses) for fast iteration signal.
- Implement the **clinical efficacy (CE)** metric: label both generated and reference reports with CheXbert, compute per-condition + micro-averaged precision/recall/F1. This is the metric that actually matters (§4).
- Error analysis: look at worst cases, check for repetition/hallucinated findings, compare your numbers against the reference paper's IU X-Ray results as a sanity check (not a target to beat).

### Week 7 (stretch) — Polish
- Beam search decoding (you already have this in `transformer/translate.py` — same pattern).
- Grad-CAM overlays tying generated phrases to image regions (nice portfolio visual).
- Small Gradio/Streamlit demo; write the README.

## 2. Image preprocessing for DenseNet121

1. **Load**: MIMIC-CXR ships as DICOM (`pydicom`) — read `pixel_array`, apply `RescaleSlope`/`RescaleIntercept`, and **invert if `PhotometricInterpretation == MONOCHROME1`**. IU X-Ray downloads are already PNG.
2. **Resize** to 224×224 (DenseNet121's ImageNet input size). Optionally decode at ~256px and random-crop to 224 for train-time augmentation.
3. **Grayscale → 3-channel**: replicate the single channel 3x (DenseNet121 expects 3-channel input).
4. **Normalize** with ImageNet stats: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` — required since you're using ImageNet-pretrained weights.
5. **Augmentation (train split only)**: small random rotation (±5–10°), slight translation/scale, mild brightness/contrast jitter. **Avoid horizontal flip** — the cardiac silhouette and organ situs are laterality-dependent, so flipping can teach the model wrong anatomy. Never vertical-flip. Skip hue/saturation jitter (grayscale image, meaningless and can distort diagnostic content).
6. **View selection**: both datasets have frontal (PA/AP) and lateral views per study. Start with **frontal-only** for v1 to keep the input simple (single image → decoder); fuse both views (e.g. two DenseNet towers concatenated into one token sequence) as a stretch goal later.
7. Optional: CLAHE/histogram equalization for contrast enhancement — common in CXR pipelines, worth an ablation but not required for v1.

## 3. CNN → Transformer bridge

- Extract from `densenet121.features(x)` (everything before the classifier) → shape `(B, 1024, 7, 7)` for a 224×224 input (DenseNet121 has stride 32 total). Apply `F.relu` after it (matches how torchvision's own `forward()` uses this layer — the pretrained weights expect that nonlinearity before pooling).
- **Don't global-average-pool.** Flatten the spatial grid instead: `(B, 1024, 7, 7) → (B, 49, 1024)`. Each of the 49 "tokens" is a spatial region's feature vector — this preserves *where* in the image a finding lives, which a single pooled vector throws away. This is exactly the same shape of problem your `transformer/model.py` encoder already solves (a sequence of vectors the decoder cross-attends over) — the CNN's 49 spatial tokens just replace the text encoder's token sequence as `context`.
- Project: `nn.Linear(1024, n_emb)` down to the decoder's embedding size, `+ LayerNorm`, `+` a learned positional embedding over the 49 grid positions (a simple `nn.Embedding(49, n_emb)` is fine for v1; a row/col-factored embedding is a nicer stretch goal).
- Feed that `(B, 49, n_emb)` tensor as `context` straight into your existing `CrossAttention`/`DecoderBlock` classes — **no changes needed there**, you're just swapping what produces `context`.
- Vocab: build a word-level vocab from the training reports with a frequency cutoff (drop tokens seen <3 times) — radiology reports are far more repetitive/templated than literary text, so expect a few hundred to a few thousand tokens, not the ~100k of your EN→FR project's `cl100k_base`. That also means the logits/cross-entropy memory blowup you hit in the translation project (dominated by `B*T*vocab_size`) mostly disappears here.
- Fine-tuning order: frozen CNN first (stable, less overfitting risk on a comparatively small paired dataset), then unfreeze only `denseblock4`+`norm5` with a smaller LR once the decoder is producing sane text (discriminative LRs: CNN << decoder).

## 4. Loss functions & metrics

- **Training loss**: token-level cross-entropy with label smoothing (`~0.1`), `ignore_index` on padding — identical to what `transformer/model.py` already does. Optional stretch: an auxiliary BCE loss from the Week 2 classifier head on the same CNN features (multi-task learning) to keep visual features clinically grounded.
- **Fast-iteration metrics**: BLEU-1..4 / ROUGE-L / CIDEr via `pycocoevalcap`. Cheap, standard, good for tracking whether training is progressing — but can be misleadingly high for generic "no acute findings" boilerplate reports, and don't verify clinical correctness.
- **Headline metric — Clinical Efficacy (CE)**: run [CheXbert](https://github.com/stanfordmlgroup/CheXbert) (BERT-based labeler, 14 conditions) on both generated and reference reports, then compute precision/recall/F1 (micro + per-condition). This is what makes the eval credible to anyone with a clinical/ML background, and is the standard practice in this literature.
- Optional further reading if you want to go beyond CE: RadGraph F1 and RadCliQ (newer, more clinically faithful metrics, referenced by the R2Gen repo below) — not required for a portfolio project, but worth name-dropping in your writeup as "aware of the limitations of CE."
- Reference point to calibrate expectations/implementation correctness (not a target to chase): **"Generating Radiology Reports via Memory-driven Transformer"** (Chen et al., EMNLP 2020) — [github.com/zhjohnchan/R2Gen](https://github.com/zhjohnchan/R2Gen). Same CNN→Transformer shape of problem, trains on both IU X-Ray and MIMIC-CXR, and its `compute_ce.py` is a working example of the CE metric computation.

## 5. Compute allocation: local RTX 5050 (8GB) vs Colab T4

| Stage | Where | Why |
|---|---|---|
| Data preprocessing / EDA | Local | CPU-bound, no GPU needed |
| (a) CNN classifier baseline | **Local RTX 5050** | DenseNet121 @224px, batch 16–32 with AMP fits comfortably in 8GB; fast local iteration |
| (b) Feature bridge + tokenizer | Local | Inference + data engineering only |
| (c) Decoder — architecture debugging (tiny overfit subset) | **Local RTX 5050** | Fast feedback loop matters more than throughput; trivial memory footprint |
| (c) Decoder — full-scale training (whole IU X-Ray, all of MIMIC-CXR) | **Colab T4 (16GB)** | 2x the VRAM headroom for bigger batches, and frees your laptop; no battery-power-cap throttling like your local GPU |
| Final CheXbert-based CE evaluation | Either | CheXbert is BERT-base — cheap enough for local GPU or even CPU on a test-set-sized batch |

Practical notes carried over from your translation project:
- Your RTX 5050 is power-capped to ~35W on battery vs. its full TDP — plug in the charger for any serious local training run.
- Colab sessions disconnect after idle/max-length limits — checkpoint to Google Drive frequently (same motivation as the OneDrive-lock retry logic in your `training.py`, different cause).

## 6. Suggested project structure

Mirror the `transformer/` project's layout — same team, same conventions, easy to navigate:

```
cancer_dettector/
  data/                    # raw + preprocessed image/report data (gitignored)
  dataloader.py            # DICOM/PNG loading, tokenizer, (image, report) dataset + collation
  cnn_model.py             # DenseNet121 classifier (stage a) + feature-extraction wrapper (stage b)
  model.py                 # bridge (projection + positional embedding) + decoder (reuse DecoderBlock/CrossAttention pattern)
  train_classifier.py      # stage (a) training loop
  training.py              # stage (c) end-to-end training loop
  generate.py              # inference/report generation CLI (beam search, like translate.py)
  requirements.txt
  README.md
```

## 7. Key references

- IU X-Ray / Open-i: <https://openi.nlm.nih.gov/>
- MIMIC-CXR: <https://physionet.org/content/mimic-cxr/2.0.0/> (credentialed access)
- R2Gen (reference architecture + CE metric code): <https://github.com/zhjohnchan/R2Gen>
- CheXbert (clinical efficacy labeler): <https://github.com/stanfordmlgroup/CheXbert>
