# TB-Sentry - Implementation Progress Log

**Project title:** Tuberculosis Screening from Chest X-Rays with Explainability

**Project name:** TB-Sentry

**Last updated:** 15 November 2026

**Project status:** Research prototype implemented; experimental evaluation and documentation completed.

---

# 1. Project Objective

TB-Sentry investigates a chest-X-ray tuberculosis screening classifier and whether quantitative spatial consistency of model explanations with available localization annotations can provide useful information for flagging predictions for human review.

The intended pipeline consists of:

1. Prepare and validate dataset manifests.
2. Train a binary TB/non-TB classifier.
3. Evaluate classification performance using class-imbalance-aware metrics.
4. Generate Grad-CAM explanations.
5. Compare heatmap locations with available TB localization annotations.
6. Investigate whether explanation consistency complements model confidence when identifying predictions for review.
7. Build a demonstration interface.

The project is an academic research prototype, not a clinically validated diagnostic system.

---

# 2. Current Scope and Terminology

The initial binary mapping is:

| Source Category | Binary Label | Meaning |
|---|---:|---|
| `health` | 0 | Non-TB-labelled |
| `sick` | 0 | Non-TB-labelled |
| `tb` | 1 | TB-labelled |

This combines the available TB-labelled categories for the initial classification experiments.

The classifier must not be described as specifically validated for active TB or as evidence of clinical generalization.

TBX11K bounding boxes are localization annotations, not pixel-level lesion masks. Heatmap-to-box measurements are described as spatial consistency with localization annotations, not proof that an explanation identifies the true clinical lesion.

The central research question remains whether explanation consistency provides useful information beyond classifier confidence. The current experiments investigate this relationship but do not establish that it improves clinical or human review decisions.

---

# 3. Dataset Preparation and Audit

The downloaded TBX11K package is stored locally under:

`data/raw/tbx11k/TBX11K/`

Dataset images and annotations are excluded from version control.

A manifest-building implementation was developed to read the official train/validation lists and corresponding JSON annotation records, map source folders to binary labels, and create an intermediate manifest.

The original manifest is:

`data/interim/tbx11k_manifest.csv`

## 3.1 Manifest Schema

The manifest contains:

- `image_path`
- `split`
- `source_category`
- `binary_label`
- `image_width`
- `image_height`
- `localization_boxes`
- `localization_categories`
- `annotation_status`

## 3.2 Original Manifest Audit Results

| Split | Total | TB-labelled | Non-TB-labelled | Images with Boxes | Images without Boxes |
|---|---:|---:|---:|---:|---:|
| Train | 6,600 | 600 | 6,000 | 599 | 6,001 |
| Validation | 1,800 | 200 | 1,600 | 200 | 1,600 |
| **Total** | **8,400** | **800** | **7,600** | **799** | **7,601** |

The training JSON contains 902 bounding boxes across 599 images.

The validation JSON contains 309 bounding boxes across 200 images.

## 3.3 Validation Checks

The following checks were performed:

- Every listed image path exists.
- Official list entries match corresponding JSON image records.
- No duplicate image paths occur across the official train and validation lists.
- Source categories are recognized.
- Binary labels follow the configured mapping.

These checks validate paths and metadata. They do not establish patient-level independence or clinical validity.

## 3.4 Class Imbalance

The training set has a 1:10 TB-labelled-to-non-TB-labelled ratio.

The validation set has a 1:8 ratio.

This imbalance motivates reporting:

- Sensitivity
- Specificity
- Precision
- F1-score
- ROC-AUC
- Average precision
- Confusion matrices

Accuracy alone is insufficient to describe performance on this imbalanced dataset.

## 3.5 Unresolved Test-Set Discrepancy

The downloaded package contains 3,302 files in its test directory, while published split information reports a 2,800-image test split.

The local `all_test.json` contains no bounding boxes.

This discrepancy remains unresolved. The downloaded test directory must not be treated as an independent labelled test set without further verification.

---

# 4. Data Loading Implementation

The project includes data-loading and preprocessing components.

Implemented components include:

- `src/tb_sentry/data/dataset.py`
- `src/tb_sentry/data/transforms.py`
- `tests/test_dataset.py`
- `scripts/test_dataloader.py`

The dataset implementation:

- Reads the CSV manifest.
- Loads images using Pillow.
- Resizes images to 224 × 224.
- Converts images to tensors.
- Applies ImageNet normalization.
- Returns image tensors and labels.
- Preserves image paths, source categories, localization boxes, localization categories, and annotation status.

## 4.1 Initial Unit Tests

The initial dataset tests covered:

1. Training dataset counts.
2. Validation dataset counts.
3. Training sample format.
4. Validation sample format.
5. Rejection of an unsupported test split.

All five initial dataset tests passed.

## 4.2 DataLoader and Model Smoke Test

The initial smoke test verified:

- Image batch shape: `(4, 3, 224, 224)`
- Label batch shape: `(4,)`
- Image dtype: `torch.float32`
- Model output shape: `(4, 2)`

The test also confirmed that variable-length localization metadata could be preserved by the custom collate function.

## 4.3 Environment Notes

Development and experimentation were performed on Windows using CPU-based PyTorch.

Pillow is used for image loading.

Earlier environment issues included a Windows application-control restriction affecting the torchvision native image extension and a compiled pandas DLL import failure. The implementation used Pillow and Python's standard-library CSV functionality rather than relying on those unavailable components.

The Windows security policy was not bypassed or disabled to accommodate the project.

---

# 5. Baseline Classification Model

A binary classification pipeline using DenseNet-121 was implemented.

Implemented components include:

- `src/tb_sentry/classification/densenet.py`
- `scripts/train_baseline.py`
- `scripts/export_validation_predictions.py`

## 5.1 Model Configuration

| Parameter | Value |
|---|---|
| Architecture | DenseNet-121 |
| Initialization | ImageNet-pretrained weights |
| Backbone | Frozen |
| Classification head | New two-class linear layer |
| Total parameters | 6,955,906 |
| Trainable parameters | 2,050 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 4 |
| Epochs | 1 |
| Random seed | 42 |
| Loss function | Class-weighted cross-entropy |
| Device | CPU |
| Initial classification threshold | 0.50 |

Class weights for `[non-TB, TB]` were approximately:

`[0.181818, 1.818182]`

The weights compensate for the imbalance in the training labels.

## 5.2 Initial Validation Results

| Metric | Result |
|---|---:|
| Validation loss | 0.15485 |
| Sensitivity | 49.50% |
| Specificity | 99.75% |
| Precision | 96.12% |
| F1-score | 0.6535 |
| ROC-AUC | 0.9599 |
| Average precision | 0.8372 |
| True negatives | 1,596 |
| False positives | 4 |
| False negatives | 101 |
| True positives | 99 |

These results come from one epoch, one seed, and the original validation split.

At threshold 0.50, the model misses 101 of the 200 TB-labelled validation images. This is an important limitation for a screening-oriented project.

The training script's `pr_auc` field stores scikit-learn average precision. Formal reporting therefore uses the term average precision (AP).

## 5.3 Generated Artifacts

- `models/checkpoints/densenet121_baseline.pt`
- `results/baseline_metrics.json`
- `results/baseline_metrics.csv`
- `results/tables/baseline_validation_predictions.csv`

The checkpoint is a local artifact and must not be committed to Git.

---

# 6. Threshold Analysis

The script `scripts/analyze_thresholds.py` evaluates thresholds from 0.05 to 0.95 in increments of 0.05 using saved validation predictions.

Generated artifacts include:

- `results/tables/baseline_threshold_analysis.csv`
- `results/figures/baseline_threshold_analysis.png`

## 6.1 Selected Results

| Threshold | Sensitivity | Specificity | Precision | F1-score | Balanced Accuracy | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.815 | 0.938 | 0.622 | 0.706 | 0.877 | 37 | 99 |
| 0.20 | 0.735 | 0.979 | 0.812 | 0.772 | 0.857 | 53 | 34 |
| 0.30 | 0.655 | 0.989 | 0.885 | 0.753 | 0.822 | 69 | 17 |
| 0.50 | 0.495 | 0.998 | 0.961 | 0.653 | 0.746 | 101 | 4 |

Threshold 0.20 produced the highest F1-score among the evaluated thresholds.

However, it was selected using the same validation set on which performance was measured. This introduces selection optimism.

Threshold 0.20 is therefore not presented as a clinically appropriate operating point or an unbiased final performance estimate.

The model's output scores have not been shown to be calibrated probabilities.

---

# 7. Exact-Duplicate Dataset Audit

An exact-duplicate audit was implemented using SHA-256 image-content hashing.

The purpose was to identify identical image content within the dataset and prevent training images from duplicating validation images.

## 7.1 Audit Procedure

The duplicate-aware manifest was generated by:

1. Calculating SHA-256 hashes for image content.
2. Identifying identical image hashes.
3. Removing training copies of images also present in validation.
4. Retaining one unique training image per content hash.
5. Preserving all validation records.

## 7.2 Audit Results

| Description | Count |
|---|---:|
| Original records | 8,400 |
| Output records | 8,280 |
| Training-validation duplicate records excluded | 38 |
| Additional training duplicate records excluded | 82 |
| Final training examples | 6,480 |
| Validation examples | 1,800 |

The generated manifest is:

`data/interim/tbx11k_manifest_deduplicated.csv`

## 7.3 Interpretation

The audit identified exact-content duplication in the original manifest.

The duplicate-aware manifest retains all 1,800 validation records while reducing the training set to 6,480 records.

SHA-256 deduplication addresses exact-content duplication. It does not establish patient-level independence or eliminate every possible near-duplicate relationship.

---

# 8. Exact-Duplicate-Aware Classification Experiment

A second DenseNet-121 experiment was conducted using the exact-duplicate-aware manifest.

## 8.1 Configuration

| Parameter | Value |
|---|---|
| Architecture | DenseNet-121 |
| Pretrained weights | ImageNet |
| Backbone | Frozen |
| Training examples | 6,480 |
| Validation examples | 1,800 |
| Seed | 42 |
| Epochs | 1 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Weighted cross-entropy |
| Threshold | 0.50 |
| Device | CPU |

## 8.2 Results

| Metric | Result |
|---|---:|
| Validation loss | 0.1583953 |
| Sensitivity | 0.7200 |
| Specificity | 0.9775 |
| Precision | 0.8000 |
| F1-score | 0.7579 |
| ROC-AUC | 0.95444 |
| Average precision | 0.84338 |
| True negatives | 1,564 |
| False positives | 36 |
| False negatives | 56 |
| True positives | 144 |

## 8.3 Comparison with Original Baseline

| Metric | Original Baseline | Exact-Duplicate-Aware |
|---|---:|---:|
| Sensitivity | 49.50% | 72.00% |
| Specificity | 99.75% | 97.75% |
| Precision | 96.12% | 80.00% |
| F1-score | 0.6535 | 0.7579 |
| ROC-AUC | 0.95985 | 0.95444 |
| Average precision | 0.83716 | 0.84338 |

The duplicate-aware experiment achieved higher sensitivity and F1-score at threshold 0.50.

Specificity and precision decreased, while ROC-AUC decreased slightly and average precision increased.

These differences do not establish that duplicate removal alone caused the performance changes or that the new model generalizes better.

## 8.4 Artifacts

- `scripts/train_duplicate_aware.py`
- `models/checkpoints/densenet121_duplicate_aware.pt`
- `results/duplicate_aware_metrics.json`
- `results/duplicate_aware_metrics.csv`
- `results/baseline_experiment_comparison.csv`

---

# 9. Targeted Near-Duplicate Exclusion Experiment

A further experiment investigated the effect of excluding two identified probable near-duplicate images from the training data.

The excluded images were:

- `data/raw/tbx11k/TBX11K/imgs/sick/s1174.png`
- `data/raw/tbx11k/TBX11K/imgs/sick/s2023.png`

## 9.1 Manifest Results

| Description | Count |
|---|---:|
| Source records | 8,280 |
| Excluded training records | 2 |
| Final records | 8,278 |
| Final training examples | 6,478 |
| Validation examples | 1,800 |

The validation records were preserved.

The resulting manifest is:

`data/interim/tbx11k_manifest_near_duplicate_excluded.csv`

## 9.2 Configuration

| Parameter | Value |
|---|---|
| Architecture | DenseNet-121 |
| Pretrained weights | ImageNet |
| Backbone | Frozen |
| Training examples | 6,478 |
| Training non-TB examples | 5,878 |
| Training TB examples | 600 |
| Validation examples | 1,800 |
| Seed | 42 |
| Epochs | 1 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Threshold | 0.50 |
| Device | CPU |

## 9.3 Results

| Metric | Result |
|---|---:|
| Validation loss | 0.1488551 |
| Sensitivity | 0.6900 |
| Specificity | 0.9850 |
| Precision | 0.8519 |
| F1-score | 0.7624 |
| ROC-AUC | 0.95393 |
| Average precision | 0.84175 |
| True negatives | 1,576 |
| False positives | 24 |
| False negatives | 62 |
| True positives | 138 |

The experiment produced a different balance of sensitivity, specificity, precision, and F1-score.

It does not demonstrate that all near-duplicate images have been identified or that patient-level leakage has been eliminated.

---

# 10. Multi-Seed Evaluation

A multiseed experiment was conducted to investigate the variation in model performance across different random seeds.

The exact-duplicate-aware training manifest was used.

## 10.1 Configuration

| Parameter | Value |
|---|---|
| Architecture | DenseNet-121 |
| Pretrained weights | ImageNet |
| Backbone | Frozen |
| Training examples | 6,480 |
| Validation examples | 1,800 |
| Seeds | 42, 123, 2026 |
| Epochs per seed | 1 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Input size | 224 × 224 |
| Threshold | 0.50 |
| Device | CPU |

## 10.2 Per-Seed Results

| Metric | Seed 42 | Seed 123 | Seed 2026 |
|---|---:|---:|---:|
| Sensitivity | 0.7800 | 0.8400 | 0.7400 |
| Specificity | 0.9544 | 0.9331 | 0.9769 |
| Precision | 0.6812 | 0.6109 | 0.8000 |
| F1-score | 0.7273 | 0.7074 | 0.7688 |
| ROC-AUC | 0.96397 | 0.95930 | 0.96117 |
| Average precision | 0.85047 | 0.84513 | 0.85644 |

## 10.3 Mean and Standard Deviation

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Sensitivity | 0.7867 | 0.0503 |
| Specificity | 0.9548 | 0.0219 |
| Precision | 0.6974 | 0.0956 |
| F1-score | 0.7345 | 0.0314 |
| ROC-AUC | 0.9615 | 0.0023 |
| Average precision | 0.8507 | 0.0057 |

ROC-AUC and average precision showed relatively small variation across the three runs.

Threshold-dependent metrics, particularly precision, showed greater variation.

The results provide preliminary evidence of seed sensitivity. Three seeds and one training epoch per seed are insufficient to establish robust generalization or convergence.

## 10.4 Artifacts

- `results/multiseed/per_seed_metrics.csv`
- `results/multiseed/`
- Individual seed metric JSON files and experiment outputs.

---

# 11. Grad-CAM Explainability Implementation

Grad-CAM was implemented to generate visual explanations for the DenseNet-121 classifier.

The analysis focuses on the TB-labelled output class.

The generated heatmaps are intended to provide a visual indication of image regions associated with the model's class score.

## 11.1 Explanation Workflow

1. Load the trained DenseNet-121 checkpoint.
2. Prepare the input chest X-ray using the model preprocessing pipeline.
3. Generate the TB-class activation map using Grad-CAM.
4. Normalize the heatmap for visualization and subsequent analysis.
5. Overlay the heatmap on the input image for visual inspection.
6. Compare activation regions with available localization annotations.

The analysis includes validation images with available bounding-box annotations.

## 11.2 Interpretation

Grad-CAM provides a post-hoc visualization of model activation patterns.

A heatmap is not a confirmed tuberculosis lesion segmentation, a clinical localization result, or proof that the model relies on medically meaningful features.

The quality and reliability of the explanations require separate evaluation.

---

# 12. Grad-CAM Spatial Consistency Evaluation

A quantitative spatial evaluation was conducted to examine the relationship between Grad-CAM activation regions and available validation bounding boxes.

## 12.1 Evaluation Dataset

| Description | Count |
|---|---:|
| Validation images with boxes | 200 |
| Total bounding boxes | 309 |
| ActiveTuberculosis boxes | 248 |
| ObsoletePulmonaryTuberculosis boxes | 61 |

The spatial evaluation used the original `tbx11k_manifest.csv` and the baseline checkpoint.

This differs from the multiseed experiments, which used the exact-duplicate-aware manifest.

## 12.2 Method

The evaluation procedure involved:

1. Selecting TB-labelled validation images with available bounding-box annotations.
2. Generating Grad-CAM heatmaps for the TB-labelled output class.
3. Resizing and normalizing heatmaps.
4. Scaling annotation coordinates to the 224 × 224 model input dimensions.
5. Thresholding normalized heatmaps at selected values.
6. Comparing activated regions with the union of available bounding boxes.

The x and y coordinates were scaled independently to match the resized image dimensions.

## 12.3 Evaluation Metrics

**Union box coverage:** Measures the proportion of annotated bounding-box area covered by the thresholded heatmap.

**Heatmap precision:** Measures the proportion of highlighted heatmap pixels that fall inside the annotated bounding-box union.

These measurements describe spatial overlap and are not clinical diagnostic metrics.

## 12.4 Results at Heatmap Threshold 0.50

| Metric | Mean | Median |
|---|---:|---:|
| Union box coverage | 20.21% | 6.94% |
| Heatmap precision | 9.88% | 2.62% |
| Mean individual box coverage | 19.78% | 6.74% |

## 12.5 Threshold Sensitivity Analysis

| Heatmap Threshold | Mean Union Box Coverage | Median Union Box Coverage | Mean Heatmap Precision |
|---|---:|---:|---:|
| 0.25 | 46.14% | 49.91% | 10.57% |
| 0.50 | 20.21% | 6.94% | 9.88% |
| 0.75 | 5.36% | 0.00% | 9.80% |

No empty heatmaps were reported at the evaluated thresholds.

## 12.6 Interpretation

Spatial overlap depends strongly on the selected heatmap threshold.

At threshold 0.50, mean union box coverage was approximately 20.21%, while mean heatmap precision was approximately 9.88%.

The relatively low median values indicate that the mean results should not be interpreted as consistent localization performance across all images.

Grad-CAM spatial overlap is an exploratory explainability measurement. It does not establish that the model has learned the correct pathology or that its predictions are clinically interpretable.

---

# 13. Confidence and Localization Association

An exploratory analysis investigated whether model confidence is associated with Grad-CAM spatial overlap with available bounding-box annotations.

## 13.1 Evaluation Setup

| Parameter | Value |
|---|---|
| Validation images | 200 |
| Association method | Spearman rank correlation |
| Bootstrap repetitions | 5,000 |
| Bootstrap random seed | 42 |

## 13.2 Results

| Spatial Metric | Spearman Correlation | 95% Bootstrap Interval |
|---|---:|---:|
| Union box coverage | 0.3014 | [0.1739, 0.4219] |
| Heatmap precision | 0.1712 | [0.0372, 0.2996] |
| Mean individual box coverage | 0.2912 | [0.1682, 0.4081] |

## 13.3 Interpretation

The observed Spearman correlations are positive, indicating a positive association between model confidence and the evaluated spatial measurements in this validation sample.

The association is stronger for union box coverage and mean individual box coverage than for heatmap precision.

These findings are exploratory. They do not demonstrate causality, incremental predictive value, or clinical validity.

The bootstrap analysis assumes that the sampled observations are suitable independent sampling units. Patient-level independence has not been established, so the intervals should be interpreted cautiously.

---

# 14. Software Testing and Demonstration Interface

A Streamlit-based demonstration interface was implemented and run successfully.

The interface provides a way to interact with the implemented image-classification workflow.

The demonstration is intended for research and educational purposes. It is not a clinical diagnostic application.

## 14.1 Automated Testing

The project test suite was executed using:

```cmd
set PYTHONPATH=src
.venv_test\Scripts\python.exe -m pytest -q
```

## 14.2 Test Results

| Item | Result |
|---|---|
| Test framework | pytest |
| Tests passed | 13 |
| Tests failed | 0 |
| Execution time | 5.78 seconds |
| Overall result | Passed |

The reported test run completed with all 13 tests passing.

This provides evidence that the tested software components behaved as expected under the available test cases. It does not establish clinical correctness or diagnostic validity.

## 14.3 Demonstration Components

The project includes:

- `app.py`
- `requirements-demo.txt`
- `run_demo.bat`

The demonstration interface was run successfully.

---

# 15. Overall Implementation Status

| Component | Status |
|---|---|
| Project structure | Implemented |
| Dataset preparation | Completed |
| Original manifest generation | Completed |
| Dataset validation | Completed |
| Data loading and preprocessing | Implemented |
| Initial dataset unit tests | Passed |
| DenseNet-121 baseline | Completed |
| Baseline evaluation | Completed |
| Threshold analysis | Completed |
| Exact-duplicate audit | Completed |
| Duplicate-aware manifest | Completed |
| Duplicate-aware classification | Completed |
| Targeted near-duplicate exclusion | Completed |
| Multiseed evaluation | Completed |
| Grad-CAM implementation | Completed |
| Spatial consistency evaluation | Completed |
| Confidence-localization association | Completed |
| Automated software testing | 13 tests passed |
| Demonstration interface | Implemented and run |
| Experiment documentation | Updated |

---

# 16. Known Limitations and Open Risks

1. **Limited training duration:** The reported classification experiments used one epoch. They do not establish convergence or optimal performance.

2. **Limited seed count:** The multiseed evaluation includes only three seeds.

3. **Frozen backbone:** The DenseNet-121 feature extractor remained frozen in the reported experiments.

4. **No independent final test evaluation:** The downloaded test-set discrepancy remains unresolved.

5. **Repeated validation use:** The validation set was used for multiple experiments and threshold analysis, introducing potential selection optimism.

6. **Patient-level independence:** Patient-level grouping and independence have not been established.

7. **Incomplete near-duplicate assessment:** Exact SHA-256 deduplication and targeted exclusion of two probable near-duplicates do not prove that every related image has been identified.

8. **Class imbalance:** The dataset contains substantially more non-TB-labelled images than TB-labelled images.

9. **Probability calibration:** Model probability calibration has not been established.

10. **Coarse localization annotations:** Bounding boxes are not pixel-level segmentation masks.

11. **Explainability limitations:** Grad-CAM visualizations and spatial overlap measurements do not prove causal faithfulness or clinical correctness.

12. **Dataset scope:** The experiments use the available TBX11K dataset and do not establish performance across independent hospitals, populations, or imaging equipment.

13. **Clinical validation:** The system has not undergone prospective clinical evaluation or clinical validation.

14. **Deployment readiness:** The demonstration interface is a research prototype and must not be treated as a medical diagnostic product.

---

# 17. Review Readiness

The current implementation supports demonstrating:

- Dataset audit and manifest generation.
- Data loading and preprocessing.
- DenseNet-121 baseline classification.
- Validation metrics and confusion matrices.
- Threshold trade-offs.
- Exact-duplicate and targeted near-duplicate audits.
- Duplicate-aware classification experiments.
- Multiseed evaluation.
- Grad-CAM visual explanations.
- Spatial consistency measurements.
- Confidence-localization association analysis.
- Automated software tests.
- The Streamlit demonstration interface.

The project can be presented as an implemented research pipeline with preliminary classification results and exploratory explainability analysis.

The results must be described as validation experiments, not independent clinical performance estimates.

---

# 18. Final Progress Statement

TB-Sentry has progressed from initial dataset preparation and baseline classification to duplicate-aware experimentation, multiseed evaluation, Grad-CAM visualization, spatial consistency analysis, confidence-association analysis, automated software testing, and a working demonstration interface.

The project demonstrates a functioning research pipeline for binary TB-labelled chest X-ray classification and exploratory explainability analysis.

The experimental results provide preliminary evidence of classification performance and measurable relationships between model activation maps and available localization annotations.

However, the current evidence does not establish clinical diagnostic performance, reliable lesion localization, patient-level generalization, or clinical deployment readiness.

**Overall progress:** Core research prototype implemented and experimentally evaluated. Experimental results and limitations documented. Further training, independent testing, patient-level validation, and clinical evaluation would be required before making stronger claims.