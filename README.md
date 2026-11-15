# TB-Sentry

### Explainable AI for Tuberculosis Screening from Chest X-Rays

**Academic Project Title:**
Tuberculosis Screening from Chest X-Rays with Explainability

**Project Type:** Final-Year B.Tech Computer Science and Engineering Capstone

**Academic Year:** 2026–2027

**Last Updated:** 15 November 2026

---

## 1. Overview

TB-Sentry is an explainable deep learning-based research prototype that investigates tuberculosis screening from chest X-ray images.

The project implements a DenseNet-121-based binary classifier to distinguish TB-labelled images from non-TB-labelled images using the TBX11K dataset.

In addition to classification, the project uses Gradient-weighted Class Activation Mapping (Grad-CAM) to generate visual explanations of model predictions.

These explanations are quantitatively compared with available tuberculosis localization bounding boxes to investigate spatial consistency.

The project also explores whether model confidence is associated with the spatial consistency of explanations.

The central research question is:

**Can the spatial consistency of Grad-CAM explanations with available TB localization annotations provide useful information alongside model confidence when examining potentially unreliable predictions?**

The investigation is exploratory. The current experiments do not establish that explanation consistency improves human-review decisions.

> **Disclaimer:** TB-Sentry is an academic research prototype. It has not been clinically validated and is not intended to diagnose tuberculosis, replace medical professionals, or guide patient treatment.

---

## 2. Key Features

- TBX11K dataset ingestion and manifest generation.
- Binary TB-labelled versus non-TB-labelled classification.
- Image preprocessing and ImageNet normalization.
- ImageNet-pretrained DenseNet-121.
- Class-weighted cross-entropy training.
- Classification evaluation and threshold analysis.
- SHA-256 exact-duplicate auditing.
- Duplicate-aware classification experiments.
- Targeted near-duplicate exclusion.
- Multiseed evaluation.
- Grad-CAM visual explanations.
- Quantitative spatial consistency analysis.
- Confidence-localization association analysis.
- Automated software tests.
- Streamlit demonstration interface.

---

## 3. Research Pipeline

```text
                  TBX11K Dataset
                         |
                         v
                Dataset Ingestion
                         |
                         v
                Manifest Generation
                         |
                         v
              Dataset Audit and Validation
                         |
                         v
                 Image Preprocessing
                         |
                         v
                 DenseNet-121 Model
                         |
                         v
                Binary Classification
                         |
                    +----+----+
                    |         |
                    v         v
                Prediction Confidence
                    |
                    v
                  Grad-CAM
                    |
                    v
              Activation Heatmap
                    |
                    v
         Spatial Consistency Evaluation
                    |
                    v
        Confidence-Localization Analysis
                    |
                    v
              Research Evaluation
```

The Streamlit interface provides a demonstration layer for the implemented classification workflow.

---

## 4. Dataset

The primary dataset used in the current experiments is TBX11K.

### Original Dataset Distribution

| Split | Total Images | TB-labelled | Non-TB-labelled |
|---|---:|---:|---:|
| Training | 6,600 | 600 | 6,000 |
| Validation | 1,800 | 200 | 1,600 |
| **Total** | **8,400** | **800** | **7,600** |

### Binary Label Mapping

| Source Category | Binary Label | Meaning |
|---|---:|---|
| `health` | 0 | Non-TB-labelled |
| `sick` | 0 | Non-TB-labelled |
| `tb` | 1 | TB-labelled |

The original dataset categories are retained in the metadata.

This mapping defines the initial experimental classification task. It does not establish clinical validity for active tuberculosis detection.

### Localization Annotations

| Split | Images with Boxes | Bounding Boxes |
|---|---:|---:|
| Training | 599 | 902 |
| Validation | 200 | 309 |

Bounding boxes are treated as coarse localization annotations, not pixel-level lesion segmentation masks.

### Dataset Handling

Raw datasets and annotations are not stored in this repository.

Users must obtain the dataset from its appropriate source and comply with the relevant usage conditions.

See [`data/README.md`](data/README.md) for dataset provenance and handling information.

### Test-Set Limitation

The downloaded TBX11K package contains 3,302 files in its test directory, while published split information reports 2,800 images.

The local `all_test.json` contains no bounding boxes.

This discrepancy remains unresolved. Therefore, the downloaded test directory is not treated as an independent labelled test set.

---

## 5. Model Architecture

The primary classifier is DenseNet-121 initialized with ImageNet-pretrained weights.

### Configuration

| Parameter | Value |
|---|---|
| Architecture | DenseNet-121 |
| Pretrained weights | ImageNet |
| Backbone | Frozen |
| Classification head | Two-class linear layer |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 4 |
| Epochs | 1 |
| Loss function | Weighted cross-entropy |
| Device | CPU |
| Initial threshold | 0.50 |

The model uses a frozen feature extractor and a trainable classification head.

Class weighting is used to account for the imbalanced training labels.

---

## 6. Experimental Results

The project includes three principal classification configurations.

All three use threshold 0.50 and the same validation set.

### Classification Comparison

| Metric | Original Baseline | Exact-Duplicate-Aware | Near-Duplicate-Excluded |
|---|---:|---:|---:|
| Sensitivity | 49.50% | 72.00% | 69.00% |
| Specificity | 99.75% | 97.75% | 98.50% |
| Precision | 96.12% | 80.00% | 85.19% |
| F1-score | 0.6535 | 0.7579 | 0.7624 |
| ROC-AUC | 0.95985 | 0.95444 | 0.95393 |
| Average precision | 0.83716 | 0.84338 | 0.84175 |

The results show different sensitivity-specificity trade-offs across the experiments.

They do not establish that duplicate removal alone caused the observed performance differences or that one configuration generalizes better.

### Threshold Analysis

The original baseline achieved its highest evaluated F1-score of approximately 0.772 at threshold 0.20.

This threshold was selected using the same validation set on which the result was measured.

It is therefore an exploratory validation result, not an unbiased final performance estimate or a clinically established operating threshold.

### Multiseed Evaluation

The duplicate-aware configuration was evaluated using seeds 42, 123, and 2026.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Sensitivity | 0.7867 | 0.0503 |
| Specificity | 0.9548 | 0.0219 |
| Precision | 0.6974 | 0.0956 |
| F1-score | 0.7345 | 0.0314 |
| ROC-AUC | 0.9615 | 0.0023 |
| Average precision | 0.8507 | 0.0057 |

The three-seed experiment provides preliminary information about variation across random seeds.

---

## 7. Explainability and Spatial Consistency

Grad-CAM is used to generate class-specific activation heatmaps for the TB-labelled output.

The generated heatmaps are compared with available validation bounding boxes.

### Spatial Consistency Results

At heatmap threshold 0.50:

| Metric | Mean | Median |
|---|---:|---:|
| Union box coverage | 20.21% | 6.94% |
| Heatmap precision | 9.88% | 2.62% |
| Mean individual box coverage | 19.78% | 6.74% |

### Confidence Association

Spearman rank correlation was used to investigate the relationship between model confidence and spatial consistency.

| Spatial Metric | Spearman Correlation |
|---|---:|
| Union box coverage | 0.3014 |
| Heatmap precision | 0.1712 |
| Mean individual box coverage | 0.2912 |

The observed associations are positive in the evaluated validation sample.

These results do not establish causality, clinical correctness, or incremental predictive value.

Grad-CAM is a post-hoc explanation method. Heatmap overlap with bounding boxes does not prove that the model has identified the true clinical lesion.

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Model architecture | DenseNet-121 |
| Image processing | Pillow |
| Numerical processing | NumPy |
| Evaluation | scikit-learn |
| Data handling | Python CSV module |
| Explainability | Grad-CAM |
| Visualization | Matplotlib |
| Demonstration interface | Streamlit |
| Testing | pytest |
| Version control | Git and GitHub |

---

## 9. Repository Structure

The project is organized into the following major components:

```text
TB-Sentry/
|
+-- data/
|   +-- raw/
|   +-- interim/
|   +-- README.md
|
+-- docs/
|   +-- methodology.md
|   +-- experiment_log.md
|   +-- progress_log.md
|   +-- project_plan.md
|   +-- meeting_notes.md
|
+-- models/
|   +-- checkpoints/
|   +-- README.md
|
+-- results/
|   +-- data_audit/
|   +-- figures/
|   +-- tables/
|   +-- multiseed/
|
+-- scripts/
|
+-- src/
|   +-- tb_sentry/
|       +-- classification/
|       +-- data/
|
+-- tests/
|
+-- app.py
+-- requirements-demo.txt
+-- run_demo.bat
+-- README.md
```

Some generated directories and artifacts are maintained locally and may not be present in a fresh clone.

---

## 10. Installation and Setup

The following instructions describe the expected local setup.

### Prerequisites

- Windows or another supported operating system.
- Python environment compatible with the installed project dependencies.
- Access to the TBX11K dataset.
- A compatible model checkpoint for the demonstration interface.

### Create a Virtual Environment

From the project root:

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate
```

### Install Project Dependencies

Install the required dependencies using the project's dependency files and the environment appropriate to the source code.

For the demonstration interface:

```cmd
pip install -r requirements-demo.txt
```

### Dataset Preparation

Obtain TBX11K separately and place it under the expected local dataset directory:

```text
data/raw/tbx11k/TBX11K/
```

The dataset must not be committed to Git.

Generate the manifest using the relevant project script after confirming that the dataset files and paths match the local installation.

---

## 11. Running the Demonstration

The project includes a Streamlit-based demonstration interface.

Before launching it, verify that the required baseline checkpoint is available:

```text
models/checkpoints/densenet121_baseline.pt
```

The checkpoint is a local artifact and is not included in the repository archive.

Install the demonstration dependencies:

```cmd
pip install -r requirements-demo.txt
```

Launch using the provided batch file:

```cmd
run_demo.bat
```

Alternatively, if the environment is configured:

```cmd
streamlit run app.py
```

The application is intended for research demonstration only.

---

## 12. Running Tests

The recorded local test execution used:

```cmd
set PYTHONPATH=src
.venv_test\Scripts\python.exe -m pytest -q
```

Recorded result:

```text
13 passed
```

The test suite verifies the implemented software components under the available test conditions.

Dataset-dependent tests require the corresponding local dataset files.

---

## 13. Documentation

Detailed project documentation is maintained in the following files:

| Document | Description |
|---|---|
| [`docs/methodology.md`](docs/methodology.md) | Research methodology and technical design |
| [`docs/experiment_log.md`](docs/experiment_log.md) | Experimental configurations, results, and limitations |
| [`docs/progress_log.md`](docs/progress_log.md) | Implementation history and project progress |
| [`docs/project_plan.md`](docs/project_plan.md) | Objectives, scope, milestones, risks, and deliverables |
| [`docs/meeting_notes.md`](docs/meeting_notes.md) | Project review preparation and meeting notes |
| [`data/README.md`](data/README.md) | Dataset provenance and handling |
| [`models/README.md`](models/README.md) | Model checkpoint and artifact information |

---

## 14. Current Implementation Status

| Component | Status |
|---|---|
| Dataset preparation | Completed |
| Manifest generation | Completed |
| Dataset validation | Completed |
| Data loading and preprocessing | Completed |
| DenseNet-121 baseline | Completed |
| Threshold analysis | Completed |
| Exact-duplicate audit | Completed |
| Duplicate-aware experiment | Completed |
| Near-duplicate exclusion experiment | Completed |
| Multiseed evaluation | Completed |
| Grad-CAM | Completed |
| Spatial consistency evaluation | Completed |
| Confidence association analysis | Completed |
| Automated testing | 13 tests passed |
| Streamlit demonstration | Implemented and run |
| Research documentation | Updated |

---

## 15. Limitations

The current research prototype has several important limitations:

1. Classification experiments used one training epoch.
2. The DenseNet-121 backbone remained frozen.
3. The multiseed evaluation contains only three seeds.
4. The validation set was reused across multiple experiments.
5. Patient-level independence has not been established.
6. The downloaded test-set discrepancy remains unresolved.
7. Exact duplicate removal does not establish complete near-duplicate removal.
8. Bounding boxes are coarse localization annotations.
9. Grad-CAM does not prove causal explanation faithfulness.
10. Model probability calibration has not been established.
11. Independent external evaluation has not been completed.
12. Clinical validation has not been performed.

These limitations must be considered when interpreting the reported results.

---

## 16. Future Work

Potential future improvements include:

- Longer training schedules.
- Additional random seeds.
- Patient-level independence verification.
- Resolution of the TBX11K test-set discrepancy.
- Independent evaluation.
- Probability calibration.
- More rigorous explanation evaluation.
- Further investigation of confidence and spatial consistency.
- External validation after establishing dataset independence.
- Clinical collaboration and evaluation.

These are future research directions and are not represented as completed features.

---

## 17. Academic and Clinical Disclaimer

TB-Sentry is an academic research prototype developed for educational and experimental purposes.

It is not a certified medical device, has not undergone clinical validation, and must not be used to diagnose tuberculosis or guide patient treatment.

The reported metrics are preliminary validation results from the available TBX11K dataset. They must not be interpreted as independent clinical performance estimates.

---

## 18. Project Status

**Status:** Core research prototype implemented and experimentally evaluated.

TB-Sentry demonstrates an integrated workflow for binary chest X-ray classification, Grad-CAM visualization, spatial consistency evaluation, confidence-association analysis, and research demonstration.

The project provides a foundation for further investigation into explainable and reliability-aware tuberculosis screening.

---

## Team

**TB-Sentry — Final-Year B.Tech Computer Science and Engineering Capstone Project**

Team members: To be updated.

---

## License

License information will be added once the project's code-sharing and publication requirements are finalized.
