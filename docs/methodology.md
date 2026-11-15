# TB-Sentry: An Explainable Deep Learning System for Tuberculosis Screening from Chest X-Rays

## 1. Introduction

TB-Sentry is an academic research prototype that investigates the use of deep learning for classifying chest X-ray images into TB-labelled and non-TB-labelled categories. The project combines image classification with Grad-CAM-based visual explanations and an exploratory spatial consistency evaluation using available localization annotations.

The research focuses not only on classification performance but also on examining whether the regions highlighted by the model overlap with annotated regions associated with tuberculosis.

The system is intended for academic experimentation and demonstration. It is not a clinically validated diagnostic system and must not be used to make medical decisions.

## 2. Research Objectives

The main objectives are:

1. Develop a chest X-ray classification pipeline using DenseNet-121.
2. Evaluate classification performance using metrics suitable for an imbalanced binary classification task.
3. Investigate the effect of exact-duplicate removal from the training data.
4. Conduct a targeted experiment excluding identified probable near-duplicate training images.
5. Examine variation in classification performance across multiple random seeds.
6. Generate Grad-CAM visual explanations for model predictions.
7. Evaluate the spatial overlap between Grad-CAM activation regions and available localization bounding boxes.
8. Explore the association between classification confidence and spatial localization metrics.

## 3. Dataset

### 3.1 Dataset Source

The project uses the TBX11K chest X-ray dataset.

The project manifest represents the dataset using training and validation splits. The original manifest contains 8,400 records:

| Split | Images |
|---|---:|
| Training | 6,600 |
| Validation | 1,800 |
| **Total** | **8,400** |

The original training split contains 600 TB-labelled images and 6,000 non-TB-labelled images. The validation split contains 200 TB-labelled images and 1,600 non-TB-labelled images.

### 3.2 Binary Label Mapping

The project maps the original dataset categories into two classification labels:

| Original Category | Binary Label | Project Interpretation |
|---|---:|---|
| health | 0 | Non-TB-labelled |
| sick | 0 | Non-TB-labelled |
| tb | 1 | TB-labelled |

The labels represent the project's dataset classification scheme. They should not be interpreted as a complete clinical distinction between healthy individuals and patients with tuberculosis.

### 3.3 Localization Annotations

The dataset provides bounding-box annotations for a subset of images.

The project uses these annotations for exploratory spatial evaluation. Bounding boxes indicate annotated regions and are not equivalent to pixel-level segmentation masks.

The presence of localization annotations does not establish that the highlighted regions represent all relevant disease findings.

## 4. Image Preprocessing

The image preprocessing pipeline converts input chest X-ray images into a consistent format for model training and inference.

The implemented preprocessing includes:

1. Loading images using the dataset loader.
2. Converting images to RGB.
3. Resizing images to 224 × 224 pixels.
4. Converting images into PyTorch tensors.
5. Normalizing image channels using ImageNet mean and standard deviation.

The normalization parameters are:

- **Mean:** [0.485, 0.456, 0.406]
- **Standard deviation:** [0.229, 0.224, 0.225]

The same image dimensions and normalization approach are used in the classification experiments.

## 5. Classification Architecture

### 5.1 DenseNet-121

The classification model is based on DenseNet-121 with ImageNet-pretrained weights.

The architecture is adapted for binary classification.

The convolutional feature-extraction backbone is frozen, while the final classification layer remains trainable.

This approach uses pretrained visual representations while limiting the number of parameters updated during training.

### 5.2 Classification Output

The model produces two class logits:

- **Class 0:** Non-TB-labelled
- **Class 1:** TB-labelled

Softmax is applied to obtain class probabilities.

The probability associated with class 1 is treated as the TB-class score.

### 5.3 Training Configuration

The principal training configuration is:

| Parameter | Configuration |
|---|---|
| Architecture | DenseNet-121 |
| Initial weights | ImageNet pretrained |
| Backbone | Frozen |
| Classification head | Trainable |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 4 |
| Epochs | 1 |
| Image dimensions | 224 × 224 |
| Execution device | CPU |
| Classification threshold | 0.5 |
| Loss function | Weighted cross-entropy |

### 5.4 Class Imbalance Handling

The dataset contains substantially fewer TB-labelled images than non-TB-labelled images.

To address this imbalance during training, inverse-frequency class weights are calculated from the training labels.

The weights are normalized to have a mean of one and supplied to the cross-entropy loss function.

This gives the minority TB-labelled class a greater contribution to the training loss.

## 6. Experimental Design

Three principal training configurations were evaluated.

### 6.1 Original Baseline Experiment

The baseline experiment uses the original dataset manifest:

`data/interim/tbx11k_manifest.csv`

Configuration:

- Random seed: 42
- Training examples: 6,600
- Validation examples: 1,800
- Epochs: 1
- Batch size: 4
- Learning rate: 0.001

The trained checkpoint is saved as:

`models/checkpoints/densenet121_baseline.pt`

Evaluation results are stored in:

- `results/baseline_metrics.json`
- `results/baseline_metrics.csv`

### 6.2 Exact-Duplicate-Aware Experiment

An exact-duplicate-aware manifest was created using SHA-256 content hashing.

The process removes duplicate training records and excludes training copies of images also present in the validation split. Validation records are retained.

The resulting manifest is:

`data/interim/tbx11k_manifest_deduplicated.csv`

The audit reports:

| Description | Count |
|---|---:|
| Original records | 8,400 |
| Output records | 8,280 |
| Training-validation duplicate records excluded | 38 |
| Additional training duplicate records excluded | 82 |
| Final training examples | 6,480 |
| Validation examples | 1,800 |

The model is trained using the same principal configuration as the original baseline.

The checkpoint is:

`models/checkpoints/densenet121_duplicate_aware.pt`

Results are stored in:

- `results/duplicate_aware_metrics.json`
- `results/duplicate_aware_metrics.csv`

### 6.3 Targeted Near-Duplicate-Excluded Experiment

A further experiment investigates two identified probable near-duplicate training images.

The two excluded images are:

- `data/raw/tbx11k/TBX11K/imgs/sick/s1174.png`
- `data/raw/tbx11k/TBX11K/imgs/sick/s2023.png`

The resulting manifest is:

`data/interim/tbx11k_manifest_near_duplicate_excluded.csv`

The final split counts are:

| Split | Images |
|---|---:|
| Training | 6,478 |
| Validation | 1,800 |
| **Total** | **8,278** |

Only the specified training-side images are excluded. The validation split remains unchanged.

The experiment uses seed 42, one epoch, batch size 4, and learning rate 0.001.

The checkpoint is:

`models/checkpoints/densenet121_near_duplicate_excluded.pt`

Results are stored in:

- `results/near_duplicate_excluded_metrics.json`
- `results/near_duplicate_excluded_metrics.csv`

Per-image validation predictions are stored in:

`results/tables/near_duplicate_excluded_validation_predictions.csv`

This is a targeted exclusion experiment. It does not demonstrate that every possible near-duplicate or patient-level overlap has been eliminated.

## 7. Multi-Seed Experiment

A multi-seed experiment was conducted to examine variation in model performance under the exact-duplicate-aware training configuration.

The experiment uses the deduplicated manifest:

`data/interim/tbx11k_manifest_deduplicated.csv`

The random seeds are:

- 42
- 123
- 2026

Each run uses:

| Parameter | Value |
|---|---:|
| Training examples | 6,480 |
| Validation examples | 1,800 |
| Epochs | 1 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Image dimensions | 224 × 224 |
| Threshold | 0.5 |

The three checkpoints are saved separately:

- `models/checkpoints/densenet121_multiseed_42.pt`
- `models/checkpoints/densenet121_multiseed_123.pt`
- `models/checkpoints/densenet121_multiseed_2026.pt`

Individual results are stored under `results/multiseed/`, with an aggregate file:

`results/multiseed/per_seed_metrics.csv`

The multi-seed experiment provides preliminary information about variation across random initializations. Because each run uses only one epoch and three seeds, it is not sufficient to establish full training convergence or robust reproducibility.

## 8. Evaluation Metrics

The classification system is evaluated using the following metrics.

### 8.1 Sensitivity

Sensitivity measures the proportion of TB-labelled validation images correctly classified as TB-labelled.

**Sensitivity = TP / (TP + FN)**

### 8.2 Specificity

Specificity measures the proportion of non-TB-labelled validation images correctly classified as non-TB-labelled.

**Specificity = TN / (TN + FP)**

### 8.3 Precision

Precision measures the proportion of images predicted as TB-labelled that are actually TB-labelled.

**Precision = TP / (TP + FP)**

### 8.4 F1-Score

The F1-score is the harmonic mean of precision and sensitivity.

**F1 = 2 × (Precision × Sensitivity) / (Precision + Sensitivity)**

### 8.5 ROC-AUC

The area under the receiver operating characteristic curve evaluates the model's ability to rank positive examples above negative examples across classification thresholds.

### 8.6 Average Precision

Average Precision summarizes the precision-recall relationship and is particularly relevant when evaluating an imbalanced classification dataset.

The project reports average precision under the names `pr_auc` or `average_precision` in different experiment outputs. These should not be assumed to represent trapezoidal integration of a precision-recall curve.

### 8.7 Classification Threshold

A probability threshold of 0.5 is used for the principal reported confusion-matrix metrics.

The threshold is an experimental operating point, not a clinically optimized decision threshold.

## 9. Explainability Using Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to generate visual explanations of the classification model.

The method uses gradients flowing into a selected convolutional feature layer to estimate the importance of feature-map channels for a target class.

The principal operations are:

1. Forward propagation of the input image.
2. Selection of the target class score.
3. Backpropagation of the target score.
4. Calculation of channel-wise gradient weights.
5. Weighted combination of feature maps.
6. Application of ReLU.
7. Resizing and normalization of the resulting heatmap.

The generated heatmap is overlaid on the input chest X-ray to visualize regions that contributed positively to the selected class score.

Grad-CAM provides a visualization of model activation. It does not prove that the model has learned clinically meaningful features or that its predictions are correct.

## 10. Spatial Consistency Evaluation

The project investigates the spatial relationship between Grad-CAM heatmaps and available localization bounding boxes.

### 10.1 Evaluation Subset

The spatial evaluation uses TB-labelled validation images with available localization boxes.

The evaluated subset contains:

- 200 images
- 309 bounding boxes

The evaluation generates Grad-CAM heatmaps for class 1.

### 10.2 Bounding-Box Processing

Bounding-box coordinates are scaled to the 224 × 224 image dimensions.

A binary annotation mask is generated by combining the available bounding boxes for each image.

The resulting mask represents the union of annotated regions.

### 10.3 Heatmap Thresholding

The normalized Grad-CAM heatmap is thresholded at 0.5 to generate a binary highlighted-region mask.

Threshold sensitivity is also investigated at 0.25 and 0.75.

### 10.4 Box Coverage

Box coverage measures the proportion of annotated bounding-box pixels covered by the thresholded Grad-CAM region.

**Box coverage = Intersection pixels / Annotated box pixels**

### 10.5 Heatmap Precision

Heatmap precision measures the proportion of highlighted Grad-CAM pixels that fall within the annotated bounding-box region.

**Heatmap precision = Intersection pixels / Highlighted heatmap pixels**

These metrics describe spatial overlap. They do not measure clinical diagnostic correctness or establish explanation faithfulness.

## 11. Confidence-Localization Association

An exploratory analysis examines the relationship between classification confidence and spatial localization metrics.

The analysis considers:

- Union box coverage
- Heatmap precision
- Mean individual box coverage

Spearman rank correlation is used to examine monotonic associations.

Bootstrap resampling is used to estimate uncertainty intervals for the correlation coefficients.

The analysis uses 200 images, 5,000 bootstrap repetitions, and random seed 42.

The results are interpreted as exploratory associations. They do not establish causality, clinical trustworthiness, or additional predictive value beyond classification confidence.

The bootstrap procedure also relies on the assumption that the sampled images are suitable independent sampling units. Patient-level independence has not been established.

## 12. Software Testing

The project includes automated tests for:

- Dataset loading and validation
- Grad-CAM functionality
- Spatial consistency calculations

The test suite previously completed with 13 passing tests in the project environment.

The tests verify implementation behavior and do not constitute clinical validation.

## 13. Limitations

The project has several important limitations:

1. Training experiments use only one epoch.
2. The multi-seed analysis uses three random seeds.
3. The validation set has been used repeatedly for model comparison and experimentation.
4. An independent external test evaluation has not been established.
5. Patient-level independence has not been established.
6. The near-duplicate exclusion experiment targets only two identified training images.
7. Bounding-box annotations are coarse localization references rather than pixel-level segmentation masks.
8. Grad-CAM heatmaps do not guarantee faithful explanations.
9. Spatial overlap metrics do not establish clinical correctness.
10. Threshold 0.5 is an experimental choice and has not been clinically optimized.
11. The project's binary label mapping is a dataset-specific research convention.
12. The system has not undergone clinical validation or prospective evaluation.

## 14. Ethical and Intended Use Statement

TB-Sentry is developed exclusively as an academic research prototype.

The system is intended to demonstrate image classification, model explainability, and exploratory spatial evaluation using chest X-ray data.

It must not be used as a replacement for qualified medical professionals, clinical assessment, radiological interpretation, or established diagnostic procedures.

Any future clinical application would require appropriate independent validation, patient-level evaluation, clinical collaboration, regulatory assessment, and careful examination of safety and fairness.

## 15. Conclusion

The methodology combines transfer learning using DenseNet-121, duplicate-aware dataset preparation, multi-seed experimentation, Grad-CAM visualization, and exploratory spatial consistency analysis.

The experiments provide an academic basis for examining classification performance and the relationship between model confidence and annotated image regions.

The findings remain preliminary and should be interpreted within the limitations of the dataset, experimental design, and validation procedure.