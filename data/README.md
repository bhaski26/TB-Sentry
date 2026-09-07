# TB-Sentry Data

This directory contains the data configuration and documentation for the
TB-Sentry project.

## Important

Medical imaging datasets are **not stored in this repository**.

Raw images, processed images, segmentation masks, bounding-box annotations,
and other dataset files must remain local and are excluded through `.gitignore`.

Users must obtain datasets from their respective official/public sources and
follow the applicable dataset licenses and usage conditions.

---

## Datasets

TB-Sentry currently considers the following publicly available chest
X-ray datasets:

1. TBX11K
2. Montgomery County Chest X-ray Dataset
3. Shenzhen Hospital Chest X-ray Dataset

Each dataset has a different role in the project.

---

## Dataset Roles

| Dataset    | Primary Role                                 | Important Annotations                     |
| ---------- | -------------------------------------------- | ----------------------------------------- |
| TBX11K     | TB classification and explanation evaluation | TB localization bounding boxes            |
| Montgomery | Lung segmentation                            | Lung segmentation masks                   |
| Shenzhen   | Exploratory/generalization analysis          | Dataset-specific radiological annotations |

The datasets must not be blindly merged because some datasets may contain
overlapping source images.

---

## TBX11K

TBX11K is the primary dataset for the TB classification and explainability
experiments.

The published TBX11K dataset is divided into:

* Training: 6,600 images
* Validation: 1,800 images
* Test: 2,800 images

TB-Sentry will preserve this published partitioning.

The availability and format of annotations for each partition will be
verified during dataset ingestion before finalizing the evaluation protocol.

TBX11K contains healthy, sick-but-non-TB, active TB, latent TB,
active+latent TB, and uncertain TB categories.

The original dataset categories must be preserved in the metadata.

For the primary binary classification task, the project will investigate
mapping the original categories into TB-positive and TB-negative groups.

The treatment of the `uncertain TB` category will be explicitly documented
and evaluated rather than silently assigning it to a class.

TBX11K provides bounding-box annotations for TB-related regions.

These bounding boxes are treated as **localization annotations** and should
not be interpreted as pixel-perfect lesion segmentation masks.

---

## Montgomery County Chest X-ray Dataset

Montgomery is primarily used for lung segmentation experiments.

The dataset contains 138 posterior-anterior chest X-rays:

* 80 normal cases
* 58 abnormal cases with manifestations consistent with TB

The dataset provides left and right lung segmentation masks, as well as
radiology readings and consensus annotations from two radiologists.

The segmentation annotations will be used to train and evaluate the proposed
lung segmentation component.

Because Montgomery may overlap with data represented in other TB datasets,
it should not automatically be treated as an independent external
classification test set.

---

## Shenzhen Hospital Chest X-ray Dataset

The Shenzhen dataset contains 662 frontal chest X-rays:

* 326 normal cases
* 336 abnormal cases with manifestations consistent with TB

The dataset is provided in PNG format and includes radiology readings.
Consensus annotations from two radiologists are also available for a subset
of 68 cases.

Shenzhen is retained as a separate dataset for exploratory and
generalization analysis.

Because Shenzhen data may overlap with TBX11K, it must not be used as an
independent classification test set after training on overlapping TBX11K
images without first establishing image-level independence.

---

## Canonical Metadata Schema

TB-Sentry uses a common internal metadata representation:

| Field            | Description                                        |
| ---------------- | -------------------------------------------------- |
| `image_id`       | Unique image identifier                            |
| `dataset`        | Source dataset                                     |
| `file_path`      | Local path to the image                            |
| `label`          | Canonical binary classification label              |
| `tb_category`    | Original dataset-specific category                 |
| `split`          | Dataset split                                      |
| `patient_id`     | Patient identifier where legitimately available    |
| `image_width`    | Original image width                               |
| `image_height`   | Original image height                              |
| `lung_mask_path` | Path to lung segmentation annotation, if available |
| `tb_bbox_path`   | Path to TB localization annotation, if available   |

Missing annotations or unavailable metadata must be represented as missing/null
rather than inferred or fabricated.

---

## Dataset-Specific Metadata

The canonical schema does not require every field to be populated for every
dataset.

### TBX11K

Expected important fields include:

* `image_id`
* `dataset`
* `file_path`
* `label`
* `tb_category`
* `split`
* `image_width`
* `image_height`
* `tb_bbox_path`

Lung segmentation masks are not assumed to be available as part of the
primary TBX11K dataset.

### Montgomery

Expected important fields include:

* `image_id`
* `dataset`
* `file_path`
* `label`
* `tb_category`
* `split`
* `image_width`
* `image_height`
* `lung_mask_path`

### Shenzhen

Expected important fields include:

* `image_id`
* `dataset`
* `file_path`
* `label`
* `tb_category`
* `split`
* `image_width`
* `image_height`

Additional annotation fields will only be populated after the downloaded
dataset is inspected and verified.

---

## Canonical Labels

TB-Sentry will maintain both:

1. The original dataset-specific category.
2. A canonical binary classification label where appropriate.

For example:

```text
Original category
       |
       v
tb_category
       |
       v
Canonical mapping
       |
       v
label = 0 / 1
```

The original category must never be discarded.

The final binary mapping will be documented in the experiment configuration
once the dataset annotations have been inspected.

This is particularly important for TBX11K because it contains multiple
TB-related categories rather than a simple binary label structure.

---

## Data Leakage Prevention

TB-Sentry follows these rules:

1. Dataset splits must be established before augmentation.
2. The published TBX11K partitioning should be preserved.
3. Test data must not be used for hyperparameter tuning.
4. Test data must not be used to select preprocessing parameters.
5. Test data must not be used to select Grad-CAM thresholds.
6. Images duplicated across datasets must not create train/test leakage.
7. Patient-level separation should be used where reliable patient identifiers
   are available.
8. Dataset source information should be retained for domain/generalization
   analysis.
9. Data augmentation must be applied only after the relevant split has been
   established.
10. Any deduplication or cross-dataset overlap analysis must be documented
    and reproducible.

---

## Dataset Overlap

TB-Sentry treats cross-dataset overlap as a potential source of evaluation
leakage.

In particular, TBX11K and the smaller public TB chest X-ray datasets should
not automatically be assumed to represent independent patient populations.

Therefore:

* TBX11K is used as the primary classification/XAI benchmark.
* Montgomery is primarily used for lung segmentation.
* Shenzhen is retained separately for exploratory and generalization analysis.
* Cross-dataset evaluation will only be described as external evaluation
  after image-level independence has been established.

---

## Data Processing Pipeline

The intended data flow is:

```text
Raw Dataset
    |
    v
Dataset-specific ingestion
    |
    v
Canonical metadata
    |
    v
Quality checks
    |
    v
Split handling
    |
    v
Preprocessing
    |
    v
Model-ready data
```

The original raw datasets should remain unchanged.

All transformations should be reproducible through project code or
configuration, subject to the terms and structure of the original datasets.

---

## Planned TB-Sentry Data Flow

The planned experimental relationship between datasets is:

```text
                    TB-Sentry DATA
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
       TBX11K        Montgomery        Shenzhen
          |               |               |
          |               v               |
          |        Lung segmentation      |
          |               |               |
          v               |               v
    Classification        |        Exploratory /
       + XAI              |        Generalization
          |               |
          v               |
    TB localization       |
    evaluation            |
          |               |
          +-------+-------+
                  |
                  v
          Research analysis
```

This architecture is subject to validation during implementation and
experimentation.

---

## Annotation Interpretation

TB-Sentry distinguishes between different annotation types.

### Lung segmentation masks

Lung masks identify the lung fields and are suitable for evaluating the
performance of the lung segmentation component.

### TB localization bounding boxes

TBX11K bounding boxes identify annotated regions associated with TB.

They provide spatial localization information but do not represent
pixel-perfect lesion segmentation.

Therefore, when Grad-CAM explanations are compared with TBX11K annotations,
the resulting metrics such as IoU or Dice will be described as measures of
**spatial consistency with available localization annotations**.

They will not be presented as definitive measurements of clinical
explanation correctness.

---

## Explainability Evaluation

The planned explainability pipeline is:

```text
CXR
 |
 v
TB Classifier
 |
 v
TB Probability
 |
 v
Grad-CAM
 |
 v
Explanation Heatmap
 |
 v
Heatmap Processing
 |
 v
Spatial Consistency
 |
 +----> IoU
 |
 +----> Dice
```

Any heatmap thresholding or post-processing parameters used to calculate
spatial consistency metrics must be selected without using the held-out test
set.

The evaluation protocol will be finalized after inspecting the available
TBX11K annotations.

---

## Data Policy

Do not commit:

* Chest X-ray images
* Patient-related information
* Segmentation masks
* TB localization annotations
* Dataset archives
* Generated model inputs
* Model checkpoints
* Generated datasets
* Temporary preprocessing outputs

The repository should contain documentation and code required to reproduce
the data preparation process, subject to the terms of the original datasets.

---

## Local Directory Structure

The expected local data organization is:

```text
data/
|
+-- README.md
|
+-- raw/
|   +-- tbx11k/
|   +-- montgomery/
|   +-- shenzhen/
|
+-- interim/
|
+-- processed/
    +-- metadata/
    +-- images/
    +-- masks/
```

The `raw/`, `interim/`, and `processed/` directories are local working
directories and are excluded from version control where appropriate.

---

## Data Integrity Principles

TB-Sentry follows these principles:

* Preserve original datasets.
* Never overwrite raw data.
* Preserve original labels and annotation semantics.
* Record dataset provenance.
* Make preprocessing reproducible.
* Keep train, validation, and test data separated.
* Avoid cross-dataset leakage.
* Do not fabricate unavailable metadata.
* Document every transformation applied to the data.
* Treat localization annotations according to their actual annotation type.

---

## Project Status

This document describes the planned data architecture and policies for
TB-Sentry.

The exact metadata fields, annotation availability, binary label mapping,
and evaluation protocol will be finalized after the official datasets have
been downloaded and programmatically inspected.
