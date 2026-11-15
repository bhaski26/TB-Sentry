# TB-Sentry - Project Plan

**Project Title:** Tuberculosis Screening from Chest X-Rays with Explainability

**Project Name:** TB-Sentry

**Academic Year:** 2026–2027

**Degree:** B.Tech - Computer Science and Engineering

**Project Type:** Academic Capstone Project

**Last Updated:** 15 November 2026

**Current Status:** Core research prototype implemented and experimentally evaluated.

---

# 1. Project Overview

TB-Sentry is an explainable deep learning-based research prototype designed to investigate tuberculosis screening from chest X-ray images.

The project uses a convolutional neural network to classify chest X-ray images into TB-labelled and non-TB-labelled categories. In addition to classification, the system generates visual explanations using Gradient-weighted Class Activation Mapping (Grad-CAM).

The project further investigates the spatial consistency of Grad-CAM activation regions with available tuberculosis localization annotations.

An additional research objective is to examine whether explanation consistency provides useful information alongside model confidence when identifying predictions that may require human review.

TB-Sentry is an academic research prototype and has not been clinically validated.

---

# 2. Problem Statement

Deep learning models have demonstrated potential for medical image classification. However, classification performance alone does not explain which image regions influence a model's prediction.

In medical imaging applications, this limitation makes it difficult to interpret model behaviour and investigate potentially unreliable predictions.

A model may produce a high-confidence classification while focusing on image regions that do not correspond to the available disease localization annotations.

TB-Sentry investigates this problem by combining:

- Deep learning-based TB classification.
- Grad-CAM visual explanations.
- Quantitative spatial consistency evaluation.
- Exploratory analysis of confidence and explanation consistency.

The objective is to study whether these components can provide a more informative research framework for examining model predictions.

---

# 3. Project Objectives

The major objectives are:

1. Develop a reproducible chest X-ray preprocessing and classification pipeline.
2. Implement binary TB-labelled versus non-TB-labelled classification using DenseNet-121.
3. Evaluate model performance using appropriate metrics for imbalanced datasets.
4. Investigate the effect of exact-duplicate removal on classification experiments.
5. Examine model performance variation across multiple random seeds.
6. Generate Grad-CAM explanations for model predictions.
7. Quantitatively compare activation regions with available localization bounding boxes.
8. Investigate associations between model confidence and spatial explanation measurements.
9. Develop a demonstration interface for the implemented classification workflow.
10. Document experimental results, limitations, and possible future improvements.

---

# 4. Research Questions

The project investigates the following questions:

**RQ1:** Can a DenseNet-121-based model distinguish TB-labelled chest X-ray images from non-TB-labelled images in the TBX11K dataset?

**RQ2:** How do classification metrics change when exact duplicate images are excluded from the training data?

**RQ3:** How much variation occurs in classification performance across different random seeds?

**RQ4:** To what extent do Grad-CAM activation regions overlap with available TB localization bounding boxes?

**RQ5:** Is model confidence associated with the spatial consistency of Grad-CAM explanations?

These questions are investigated experimentally. The current results do not establish clinical validity or prove that explanation consistency improves human-review decisions.

---

# 5. Project Scope

## 5.1 Included Scope

The implemented project includes:

- TBX11K dataset ingestion.
- Dataset manifest generation.
- Binary label mapping.
- Image preprocessing.
- DenseNet-121 classification.
- Weighted cross-entropy loss.
- Baseline model evaluation.
- Threshold analysis.
- Exact-duplicate auditing.
- Duplicate-aware classification.
- Targeted near-duplicate exclusion.
- Multiseed evaluation.
- Grad-CAM visualization.
- Spatial consistency evaluation.
- Confidence-localization association analysis.
- Automated software testing.
- Streamlit demonstration interface.

## 5.2 Excluded or Unverified Scope

The following are not established as completed or validated capabilities:

- Clinical diagnosis of tuberculosis.
- Prospective clinical evaluation.
- Patient-level independent validation.
- Independent external test performance.
- Pixel-level lesion segmentation.
- Clinically validated decision thresholds.
- Probability calibration.
- Proven improvement in human-review decisions.
- Production deployment in healthcare environments.

Lung segmentation using U-Net and CLAHE preprocessing were considered as possible extensions but are not part of the verified experimental pipeline.

---

# 6. Dataset Plan

The primary dataset is TBX11K.

## 6.1 Original Dataset Distribution

| Split | Total Images | TB-labelled | Non-TB-labelled |
|---|---:|---:|---:|
| Training | 6,600 | 600 | 6,000 |
| Validation | 1,800 | 200 | 1,600 |
| **Total** | **8,400** | **800** | **7,600** |

## 6.2 Binary Label Mapping

| Source Category | Binary Label |
|---|---:|
| health | 0 |
| sick | 0 |
| tb | 1 |

The original categories are retained in the dataset metadata.

## 6.3 Localization Annotations

| Split | Images with Boxes | Number of Boxes |
|---|---:|---:|
| Training | 599 | 902 |
| Validation | 200 | 309 |

Bounding boxes are treated as coarse localization annotations rather than pixel-level lesion masks.

## 6.4 Data Leakage Prevention

The project includes:

- SHA-256 exact-content hashing.
- Removal of exact duplicates across training and validation.
- Removal of additional duplicate training records.
- Targeted exclusion of two identified probable near-duplicate training images.
- Preservation of validation records during the documented manifest transformations.

Patient-level independence has not been established.

The downloaded test-set discrepancy remains unresolved, so the local test directory is not treated as an independent labelled test set.

---

# 7. Proposed System Architecture

The implemented research workflow is:

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
            +------+------+
            |             |
            v             v
      Prediction       Confidence
            |
            v
          Grad-CAM
            |
            v
     Activation Heatmap
            |
            v
   Spatial Consistency Analysis
            |
            v
  Confidence-Localization Analysis
            |
            v
      Research Evaluation
```

---

# 8. Methodology

## 8.1 Experimental Environment

The project is implemented in Python using PyTorch for model development and evaluation. The workflow is designed to be reproducible through fixed random seeds, recorded experiment configurations, and documented dataset-manifest transformations.

The verified baseline configuration is:

| Component | Configuration |
|---|---|
| Model | DenseNet-121 |
| Initialisation | ImageNet-pretrained weights |
| Backbone | Frozen |
| Trainable component | Two-class linear classifier |
| Input size | 224 × 224 |
| Normalisation | ImageNet mean and standard deviation |
| Loss | Weighted cross-entropy |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Batch size | 4 |
| Training duration | 1 epoch for the documented baseline and multiseed runs |
| Execution device | CPU |
| Baseline random seed | 42 |

These settings describe the verified experiments and should not be interpreted as an optimised clinical model configuration.

## 8.2 Image Preprocessing

Chest X-ray images are loaded from the dataset and transformed to the model input dimensions of 224 × 224 pixels. ImageNet normalisation is applied to provide inputs compatible with the pretrained DenseNet-121 backbone.

The same documented preprocessing convention is used when generating model predictions and Grad-CAM explanations for the corresponding experiment.

## 8.3 Model Training

DenseNet-121 is initialised with ImageNet-pretrained parameters. The convolutional backbone is frozen, and the final classification layer is configured for two output classes.

Weighted cross-entropy is used to account for the imbalance between TB-labelled and non-TB-labelled samples. The classifier is trained using Adam with a learning rate of 0.001.

The baseline and the documented multiseed experiments use one training epoch. The limited training duration is an important constraint when interpreting the results.

## 8.4 Evaluation

The model produces class scores that are converted into predictions using a decision threshold. Evaluation includes confusion-matrix counts and threshold-dependent metrics, together with threshold-independent ranking metrics.

The validation set is used for the documented experiments. Because threshold selection and evaluation use the same validation set in the baseline threshold analysis, the selected-threshold results are exploratory and are not an unbiased final performance estimate.

## 8.5 Explainability

Grad-CAM is used to generate class-discriminative activation heatmaps. The heatmaps provide a visual indication of image regions that contribute to the selected model output.

The heatmaps are compared with the available TB localization bounding boxes. These annotations are treated as coarse reference regions; they are not pixel-level ground-truth masks.

## 8.6 Spatial Consistency Analysis

Spatial consistency is assessed by comparing thresholded heatmap activation with the union of available localization boxes. The analysis includes heatmap coverage within annotated regions and the proportion of activated heatmap area falling within those regions.

The analysis is repeated at multiple heatmap thresholds to examine sensitivity to the activation threshold. Associations between model confidence and spatial measurements are explored using Spearman rank correlation and bootstrap confidence intervals.

The spatial analysis is exploratory. It does not establish that Grad-CAM identifies lesions correctly or that a higher overlap necessarily represents a clinically better explanation.

---

# 9. Experimental Plan

The experiments are organised to answer the research questions while keeping the data transformations and limitations explicit.

| Experiment | Purpose | Main comparison or output |
|---|---|---|
| Baseline classification | Establish initial DenseNet-121 performance | Validation metrics at threshold 0.5 |
| Threshold analysis | Examine the effect of the classification threshold | Sensitivity, specificity, precision and F1 across thresholds |
| Exact-duplicate audit | Identify exact image-content duplication | SHA-256 duplicate groups and cleaned manifests |
| Duplicate-aware classification | Examine performance after exact-duplicate handling | Validation metrics on the documented duplicate-aware training manifest |
| Targeted near-duplicate exclusion | Explore the effect of excluding two suspected near-duplicate training records | Comparison with the duplicate-aware experiment |
| Multiseed evaluation | Examine run-to-run variation | Metrics for seeds 42, 123 and 2026 |
| Grad-CAM generation | Produce visual explanations | Activation heatmaps for validation images |
| Spatial consistency evaluation | Compare activation with available boxes | Coverage and heatmap precision at selected heatmap thresholds |
| Confidence-localization analysis | Explore association between confidence and spatial measurements | Spearman correlations and bootstrap intervals |

## 9.1 Dataset Manifest Versions

The original TBX11K training and validation distributions contain 6,600 and 1,800 images respectively.

The exact-duplicate audit produced a duplicate-aware manifest with 6,480 training images and 1,800 validation images. The audit excluded 38 training-validation duplicate records and 82 additional duplicate training records.

A targeted follow-up excluded two additional training images, `sick/s1174.png` and `sick/s2023.png`, producing a training manifest of 6,478 images while retaining the 1,800 validation records.

These manifest versions are distinct experimental conditions and should be identified explicitly when reporting results.

## 9.2 Test-Set Policy

The downloaded test directory contains 3,302 files, whereas the published dataset description indicates 2,800 test images. The available `all_test.json` does not provide bounding boxes.

The discrepancy has not been resolved. Consequently, the local test directory is not used as an independent labelled test set, and no independent test-set performance claim is made.

---

# 10. Evaluation Metrics

The project uses metrics suited to binary classification and class imbalance.

| Metric | Purpose |
|---|---|
| True positives (TP) | TB-labelled images correctly predicted as TB-labelled |
| True negatives (TN) | Non-TB-labelled images correctly predicted as non-TB-labelled |
| False positives (FP) | Non-TB-labelled images predicted as TB-labelled |
| False negatives (FN) | TB-labelled images predicted as non-TB-labelled |
| Sensitivity / Recall | Measures the proportion of TB-labelled validation images detected |
| Specificity | Measures the proportion of non-TB-labelled validation images correctly rejected |
| Precision | Measures the proportion of positive predictions that are TB-labelled |
| F1-score | Harmonic mean of precision and sensitivity |
| ROC-AUC | Measures ranking discrimination across classification thresholds |
| Average Precision (AP) | Summarises the precision-recall curve |

For spatial analysis:

- **Union box coverage** measures the proportion of the annotated union region covered by thresholded activation.
- **Heatmap precision** measures the proportion of activated heatmap area that falls within the annotated union region.
- **Mean individual box coverage** evaluates activation coverage for individual annotated boxes before averaging.

Metric values depend on the experiment, manifest, model checkpoint, and threshold. Results from different conditions must not be combined as though they were produced by one identical evaluation.

---

# 11. Experimental Results Summary

The following values are the recorded results available for this project plan. They are research-prototype results on the documented validation data, not clinical performance claims.

## 11.1 Baseline Classification

The baseline model achieved the following results at a classification threshold of 0.5:

| Metric | Result |
|---|---:|
| Validation loss | 0.1548466 |
| Sensitivity | 0.4950 |
| Specificity | 0.9975 |
| Precision | 0.961165 |
| F1-score | 0.653465 |
| ROC-AUC | 0.959853 |
| Average Precision | 0.8371597 |
| TN | 1,596 |
| FP | 4 |
| FN | 101 |
| TP | 99 |

The baseline operating point has high specificity and precision but detects fewer than half of the TB-labelled validation images at the selected threshold. This illustrates why sensitivity and threshold behaviour must be examined rather than relying on accuracy or ROC-AUC alone.

## 11.2 Threshold Analysis

A threshold sweep identified a validation F1-score of approximately 0.772 at a threshold of 0.20.

| Metric | Result at threshold 0.20 |
|---|---:|
| Sensitivity | 0.735 |
| Specificity | 0.979 |
| Precision | 0.812 |
| F1-score | 0.772 |
| FN | 53 |
| FP | 34 |

The threshold was selected using the same validation data on which these results were evaluated. Therefore, this result is exploratory and may be optimistic. The threshold is not a clinically validated operating point.

## 11.3 Exact-Duplicate-Aware Experiment

The duplicate-aware experiment recorded:

| Metric | Result |
|---|---:|
| Validation loss | 0.1583953 |
| Sensitivity | 0.7200 |
| Specificity | 0.9775 |
| Precision | 0.8000 |
| F1-score | 0.7578947 |
| ROC-AUC | 0.9544375 |
| Average Precision | 0.8433799 |
| TN | 1,564 |
| FP | 36 |
| FN | 56 |
| TP | 144 |

This experiment provides a comparison with the baseline under the documented duplicate-aware training manifest. It does not establish patient-level independence.

## 11.4 Targeted Near-Duplicate Exclusion

After excluding the two targeted training images, the recorded results were:

| Metric | Result |
|---|---:|
| Validation loss | 0.1488551 |
| Sensitivity | 0.6900 |
| Specificity | 0.9850 |
| Precision | 0.8518519 |
| F1-score | 0.7624309 |
| ROC-AUC | 0.953925 |
| Average Precision | 0.8417547 |
| TN | 1,576 |
| FP | 24 |
| FN | 62 |
| TP | 138 |

The change in metrics is descriptive of this targeted experiment. Excluding two suspected near-duplicates does not prove that all near-duplicate images have been identified or removed.

## 11.5 Multiseed Evaluation

Three random seeds were evaluated using the exact-duplicate-aware training manifest, with 6,480 training images and 1,800 validation images.

| Seed | Sensitivity | Specificity | Precision | F1-score | ROC-AUC | AP |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7800 | 0.954375 | 0.6812227 | 0.7272727 | 0.9639719 | 0.8504675 |
| 123 | 0.8400 | 0.933125 | 0.6109091 | 0.7073684 | 0.9593031 | 0.8451343 |
| 2026 | 0.7400 | 0.976875 | 0.8000000 | 0.7688312 | 0.9611719 | 0.8564404 |
| **Mean** | **0.7867** | **0.9548** | **0.6974** | **0.7345** | **0.9615** | **0.8507** |
| **Standard deviation** | **0.0503** | **0.0219** | **0.0956** | **0.0314** | **0.00235** | **0.00566** |

The multiseed results show variation in threshold-dependent metrics, particularly precision. Only three seeds and one training epoch were used; these results are preliminary and do not establish robust generalisation.

## 11.6 Grad-CAM Spatial Consistency

The spatial evaluation used the original TBX11K manifest and the baseline checkpoint. It covered 200 validation images with 309 localization boxes:

- ActiveTuberculosis: 248 boxes.
- ObsoletePulmonaryTuberculosis: 61 boxes.

At a heatmap threshold of 0.5:

| Measurement | Mean | Median |
|---|---:|---:|
| Union box coverage | 20.21% | 6.94% |
| Heatmap precision | 9.88% | 2.62% |
| Mean individual box coverage | 19.78% | 6.74% |

### Heatmap threshold sensitivity

| Heatmap threshold | Mean union box coverage | Median union box coverage | Mean heatmap precision |
|---:|---:|---:|---:|
| 0.25 | 46.14% | 49.91% | 10.57% |
| 0.50 | 20.21% | 6.94% | 9.88% |
| 0.75 | 5.36% | 0.00% | 9.80% |

No empty heatmaps were recorded in this evaluation.

The spatial evaluation used a different data-manifest/checkpoint condition from the duplicate-aware multiseed experiments. These results must therefore be reported as a separate analysis rather than merged with the multiseed results.

## 11.7 Confidence and Localization Associations

Spearman rank correlations were calculated across 200 validation images, with 5,000 bootstrap repetitions and random seed 42.

| Spatial measurement | Spearman rho | Bootstrap 95% interval |
|---|---:|---:|
| Union box coverage | 0.3014067 | [0.1738773, 0.4219396] |
| Heatmap precision | 0.1712265 | [0.0371628, 0.2995818] |
| Mean individual box coverage | 0.2911938 | [0.1681919, 0.4080935] |

These are exploratory associations. They do not demonstrate causality, clinical correctness, or improved human-review outcomes. The bootstrap interpretation is also limited by the absence of established patient-level independence.

---

# 12. Software Design and Demonstration

## 12.1 Main Components

The project is organised around the following functional components:

1. Dataset ingestion and manifest generation.
2. Dataset auditing and duplicate analysis.
3. Image preprocessing and model input preparation.
4. DenseNet-121 classification.
5. Validation and threshold-based evaluation.
6. Grad-CAM generation.
7. Spatial consistency measurement.
8. Confidence-localization association analysis.
9. Automated tests.
10. Streamlit demonstration interface.

The implementation should be described according to the actual source files in the repository. File names and module boundaries may evolve during development.

## 12.2 Demonstration Interface

A Streamlit interface is available through `app.py`. The demo is intended to demonstrate the implemented research workflow and should not be represented as a medical diagnostic product.

The repository also includes:

- `requirements-demo.txt` for the demonstration dependencies.
- `run_demo.bat` as a Windows launch helper.

The local model checkpoint is stored at:

`models/checkpoints/densenet121_baseline.pt`

The checkpoint is a local model artifact and is excluded from version control by the project's ignore rules.

## 12.3 Demonstration Safety Statement

The interface and any presentation of its output should clearly state that:

- The system is an academic research prototype.
- It is not intended for diagnosis or treatment decisions.
- Predictions may be incorrect.
- Grad-CAM heatmaps are model explanations, not confirmed lesion maps.
- A qualified healthcare professional must make clinical decisions.

---

# 13. Testing and Quality Assurance

## 13.1 Automated Testing

The project has an automated test suite. The latest recorded local test run completed successfully:

```text
13 passed in 6.93s
```

The tests were run from the project directory using:

```cmd
set PYTHONPATH=src
.venv_test\Scripts\python.exe -m pytest -q
```

This confirms that the 13 tests passed in that local environment at the time of the recorded run. It does not establish clinical validity or guarantee that every possible input and deployment environment is covered.

## 13.2 Data Quality Checks

The documented data checks include:

- Original split counts and class distribution.
- Binary label mapping.
- Localization annotation counts.
- SHA-256 exact-content duplicate auditing.
- Training-validation duplicate exclusion.
- Additional training duplicate exclusion.
- Targeted review of two probable near-duplicate records.
- Documentation of the unresolved test-directory discrepancy.

## 13.3 Reproducibility

For each reported experiment, the project should retain or document:

- Dataset manifest version.
- Model configuration.
- Random seed.
- Training settings.
- Checkpoint identity.
- Evaluation threshold.
- Metric outputs.
- Relevant analysis scripts.

Results from different manifest versions or checkpoints should be labelled separately.

## 13.4 Manual Demonstration Checks

Before a project review, verify that:

1. The application launches from the documented command or batch file.
2. A valid chest X-ray image can be selected.
3. The model returns a prediction without an application error.
4. The displayed confidence is clearly labelled as model output.
5. The Grad-CAM explanation is displayed when supported by the application.
6. The research-prototype disclaimer is visible.
7. Missing or invalid input is handled gracefully.

---

# 14. Project Risks and Mitigation

| Risk | Potential impact | Mitigation or reporting approach |
|---|---|---|
| Class imbalance | A model may favour the majority class | Use class-weighted loss and report sensitivity, specificity, precision and F1 |
| Exact duplicate images | Inflated or misleading evaluation | Perform SHA-256 auditing and document manifest transformations |
| Undetected near-duplicates | Residual similarity or leakage risk | Report the limits of exact hashing and targeted near-duplicate review |
| Unknown patient-level independence | Validation results may not represent independent patients | Explicitly disclose that patient-level independence has not been established |
| Test-set discrepancy | Independent test evaluation cannot be trusted | Do not report the local test directory as an independent labelled test set |
| Limited training duration | Model may be undertrained | State that the documented runs used one epoch and treat results as preliminary |
| Small number of random seeds | Limited evidence about run-to-run variability | Report all three seeds and avoid broad robustness claims |
| Threshold selection on validation data | Optimistic selected-threshold estimates | Label threshold-sweep results exploratory; require a separate validation protocol for future work |
| Coarse localization boxes | Spatial overlap may not reflect lesion boundaries | Treat boxes as coarse annotations, not segmentation masks |
| Grad-CAM limitations | Heatmaps may not faithfully identify causal image evidence | Present heatmaps as qualitative/exploratory explanations |
| Lack of external validation | Unknown performance outside the dataset | Avoid claims of clinical generalisation |
| Medical-use misinterpretation | Users may treat a prototype as a diagnosis | Display a clear non-diagnostic disclaimer |

---

# 15. Limitations and Future Work

## 15.1 Current Limitations

The current project has the following important limitations:

1. The model has been evaluated on the documented validation split; an independent labelled test evaluation has not been established.
2. Patient-level independence between training and validation data is unknown.
3. The local test-directory count differs from the published count, and the discrepancy remains unresolved.
4. The baseline and multiseed training runs used only one epoch.
5. The multiseed experiment includes only three random seeds.
6. Threshold selection and evaluation on the same validation set may produce optimistic estimates.
7. Exact duplicate removal does not eliminate every possible near-duplicate or patient-level overlap.
8. The Grad-CAM spatial evaluation uses coarse bounding-box annotations rather than pixel-level masks.
9. Spatial measurements depend on the heatmap threshold.
10. The confidence-localization analysis is observational and does not establish causation.
11. No probability calibration or clinically validated decision threshold has been established.
12. No prospective clinical evaluation or external clinical validation has been performed.

## 15.2 Future Work

Potential future research directions include:

- Resolve the test-set discrepancy using authoritative dataset documentation.
- Establish patient-level grouping and leakage checks if patient identifiers are available.
- Evaluate on an independently defined and appropriately labelled test set.
- Increase training duration and conduct controlled hyperparameter experiments.
- Expand multiseed evaluation to improve the assessment of variability.
- Investigate alternative architectures and fine-tuning strategies.
- Evaluate calibration and decision-threshold selection using a properly separated validation protocol.
- Investigate more systematic near-duplicate detection.
- Compare Grad-CAM with additional explanation methods.
- Evaluate explanation stability under controlled image transformations.
- Study localization using stronger annotation sources where available.
- Conduct external and clinical evaluation under appropriate ethical and institutional oversight.
- Assess human-review utility through a properly designed study before making claims about decision support.

These are proposed directions and should not be presented as completed work.

---

# 16. Deliverables, Milestones, and Review Readiness

## 16.1 Project Deliverables

The project deliverables include:

| Deliverable | Current status |
|---|---|
| TBX11K ingestion and manifest workflow | Implemented |
| Binary label mapping | Implemented |
| DenseNet-121 baseline classification | Implemented and evaluated |
| Exact-duplicate audit | Completed for documented manifests |
| Duplicate-aware experiment | Completed |
| Targeted near-duplicate experiment | Completed |
| Three-seed evaluation | Completed as a preliminary experiment |
| Grad-CAM visualization | Implemented |
| Spatial consistency analysis | Completed as an exploratory analysis |
| Confidence-localization association analysis | Completed as an exploratory analysis |
| Automated test suite | 13 tests passing in the latest recorded run |
| Streamlit demonstration | Present; perform a fresh pre-review launch check |
| Project documentation | Methodology, experiment log, progress log, project plan, meeting notes and README maintained |
| Independent clinical validation | Not completed |
| Independent labelled test-set evaluation | Not established |

## 16.2 Review Preparation Priorities

Before the academic review:

1. Verify the project launches in the intended local environment.
2. Confirm the required checkpoint is present locally.
3. Run the automated test suite and retain the output.
4. Prepare a short demonstration using a permitted sample image.
5. Present the dataset distribution and label mapping.
6. Explain the baseline and duplicate-aware experiment conditions separately.
7. Explain why sensitivity and specificity change with the decision threshold.
8. Demonstrate Grad-CAM and explain the limitations of bounding-box overlap.
9. Present the multiseed results as preliminary.
10. Clearly state the unresolved test-set and patient-independence limitations.
11. Avoid describing the system as clinically validated or as a diagnostic device.

The department's review date and any final submission requirements should be confirmed against the latest official department communication. This plan does not independently establish the institution's review schedule.

## 16.3 Review Presentation Structure

A concise review presentation may follow this order:

1. Title and project motivation.
2. Problem statement and research questions.
3. Dataset description and class distribution.
4. Data policy, duplicate audit and leakage limitations.
5. DenseNet-121 architecture and training configuration.
6. Baseline results and threshold analysis.
7. Duplicate-aware and multiseed experiments.
8. Grad-CAM explanation examples.
9. Spatial consistency measurements and confidence associations.
10. Software demonstration.
11. Limitations, risks and future work.
12. Conclusion.

## 16.4 Overall Project Status

TB-Sentry has progressed from dataset preparation to an implemented and experimentally evaluated research prototype. The current work includes a DenseNet-121 classification baseline, duplicate-aware experiments, preliminary multiseed evaluation, Grad-CAM explanations, quantitative spatial consistency analysis, a demonstration interface, automated tests, and supporting documentation.

The project provides an experimental framework for studying classification and explanation behaviour on TBX11K. Its findings remain limited by the available validation protocol, unresolved patient-level independence, the test-directory discrepancy, short training duration, limited number of random seeds, and the coarse nature of localization annotations.

**Final status:** Core research prototype implemented and experimentally evaluated. Clinical validation, independent external testing, and deployment for medical decision-making are outside the scope of the verified results.