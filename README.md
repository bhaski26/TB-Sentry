# TB-Sentry

### Explainable AI for Tuberculosis Screening from Chest X-Rays

**Academic Project Title:**

**TUBERCULOSIS SCREENING FROM CHEST X-RAYS WITH EXPLAINABILITY**

TB-Sentry is an academic research prototype that investigates deep learning-based tuberculosis screening from chest X-ray images with interpretable model predictions.

The system explores not only whether a chest X-ray is classified as suggestive of tuberculosis, but also **where the model focuses when making its prediction** and whether the generated explanation is spatially consistent with available tuberculosis-related localization annotations.

The project also investigates whether explanation consistency can complement prediction confidence when identifying cases that may require human review.

> **Disclaimer:** TB-Sentry is an academic research prototype. It has not been clinically validated and is not intended to diagnose tuberculosis or replace qualified medical professionals.

---

## Proposed Pipeline

```text
Chest X-ray
     |
     v
Preprocessing
     |
     v
Optional Lung Segmentation
     |
     v
TB Classification
     |
     +-----------> TB Prediction Score
     |
     v
Grad-CAM Explanation
     |
     v
Spatial Consistency Evaluation
     |
     v
Confidence + Explanation Consistency
     |
     v
Experimental Human-Review Signal
```

The pipeline is a research design and will be refined according to experimental results and available computational resources.

Lung segmentation is currently a proposed component, not part of the implemented baseline.

---

## Proposed Models and Methods

| Component | Proposed Approach | Current Status |
|---|---|---|
| Image loading | Pillow | Implemented |
| Image preprocessing | Resize to 224 × 224 and ImageNet normalization | Implemented |
| CLAHE | Contrast enhancement | Proposed; not yet evaluated |
| Lung segmentation | U-Net | Proposed; not yet implemented |
| TB classification | DenseNet-121 | Preliminary baseline implemented |
| Explainability | Grad-CAM | Pending |
| Explanation evaluation | Spatial consistency with localization boxes | Pending |
| Reliability analysis | Prediction confidence and explanation consistency | Pending |
| Demonstration interface | Streamlit or equivalent | Planned |

The implemented classifier currently uses ImageNet-pretrained DenseNet-121 with a frozen backbone and a newly initialized binary classification head.

Alternative architectures and preprocessing methods may be evaluated if justified by the experiments and available resources.

---

## Datasets

The project investigates publicly available chest X-ray datasets, including:

- Shenzhen Hospital Chest X-ray Dataset
- Montgomery County Chest X-ray Dataset
- TBX11K

Additional datasets may be considered for pretraining or external evaluation if their provenance, annotations, and suitability can be established.

### Current Dataset

The initial experiments use the official TBX11K training and validation lists.

| Split | Images | TB-labelled | Non-TB-labelled |
|---|---:|---:|---:|
| Training | 6,600 | 600 | 6,000 |
| Validation | 1,800 | 200 | 1,600 |
| Total | 8,400 | 800 | 7,600 |

For the initial binary classification experiment, the `tb` source category is mapped to label `1`, while `health` and `sick` are mapped to label `0`.

This is the project's initial experimental label mapping. It should not be interpreted as evidence that the model is clinically validated for detecting active tuberculosis or generalizes to other populations.

TBX11K localization boxes are available for only a subset of images. They are coarse localization annotations, not pixel-level lesion masks.

**Dataset policy:** Raw datasets and annotations are not stored directly in this repository.

See [`data/README.md`](data/README.md) for data provenance and handling information.

---

## Research Focus

The primary research focus is **explainable and reliability-aware tuberculosis screening**.

Rather than treating a Grad-CAM heatmap as proof that a prediction is trustworthy, TB-Sentry will investigate:

1. Whether the classifier can distinguish TB-labelled images from non-TB-labelled images.
2. Whether Grad-CAM explanations are spatially consistent with available TB localization annotations.
3. Whether explanation consistency provides useful information beyond prediction confidence when identifying potentially unreliable cases.
4. Whether a combined confidence-and-explanation approach can be evaluated as a human-review signal.

The central research question is exploratory. It has not yet been established that explanation consistency improves review decisions.

Because bounding boxes do not provide pixel-perfect lesion boundaries, the explanation evaluation will be described as **spatial consistency with localization annotations**, rather than proof of clinical correctness.

---

## Current Implementation Status

**Last updated:** 12 October 2026

### Completed

- TBX11K training and validation manifest generation.
- Dataset metadata and path validation.
- Pillow-based dataset loading.
- Image resizing and ImageNet normalization.
- Dataset unit tests: five passed.
- DataLoader and model smoke test.
- ImageNet-pretrained DenseNet-121 baseline with a frozen backbone.
- One-epoch preliminary training experiment.
- Per-image validation prediction export.
- Classification threshold analysis.

### Preliminary Baseline Results

The following results were obtained on the current validation set after one training epoch.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9599 |
| Average precision | 0.8372 |
| Sensitivity at threshold 0.50 | 49.5% |
| Specificity at threshold 0.50 | 99.75% |
| Precision at threshold 0.50 | 96.12% |
| F1-score at threshold 0.50 | 0.6535 |

At threshold `0.50`, the model missed 101 of the 200 TB-labelled validation images.

A threshold sweep produced an F1-score of `0.772` at threshold `0.20`. This threshold was selected using the same validation set, so the result is exploratory and is not an unbiased final performance estimate or a clinically established operating point.

The baseline is preliminary: it uses one training epoch, one random seed, and a frozen backbone.

### Pending Work

- Grad-CAM implementation and qualitative validation.
- Quantitative spatial-consistency evaluation using available TBX11K localization boxes.
- Investigation of whether explanation consistency complements prediction confidence.
- Probability calibration.
- Further investigation of patient-level independence and duplicate image content.
- Resolution of the downloaded test-set discrepancy.
- Additional model training and reproducibility experiments.
- Demonstration interface, subject to completion of the core research pipeline.

No claim has yet been established that explanation consistency improves human-review decisions.

---

## Repository Documentation

Detailed records of implementation and experimentation are maintained in:

- [`docs/progress_log.md`](docs/progress_log.md) — Implementation progress, dataset audit, limitations, and next steps.
- [`docs/experiment_log.md`](docs/experiment_log.md) — Dataset audit and baseline experiment results.
- [`docs/methodology.md`](docs/methodology.md) — Research methodology.
- [`docs/project_plan.md`](docs/project_plan.md) — Project plan.
- [`data/README.md`](data/README.md) — Dataset provenance and data handling policy.

---

## Project Status

**Active Research and Development**

TB-Sentry is an academic research prototype. Its preliminary results must not be interpreted as evidence of clinical performance or suitability for medical decision-making.

---

## Team

**TB-Sentry — Final-Year B.Tech CSE Capstone Project**

Team members: *To be added*

---

## License

License information will be added once the project's code-sharing and publication requirements are finalized.