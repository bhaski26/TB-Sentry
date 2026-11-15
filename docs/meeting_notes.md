# TB-Sentry - Project Review Meeting Notes

**Project Title:** Tuberculosis Screening from Chest X-Rays with Explainability

**Project Name:** TB-Sentry

**Academic Year:** 2026–2027

**Department:** Computer Science and Engineering

**Institution:** Dayananda Sagar University

**Document Date:** 15 November 2026

**Purpose:** Project review preparation and implementation status record.

---

# 1. Review Schedule

The current preparation checkpoint is 16 November 2026.

The university notice provided for reference states that Capstone Project Review 1 is scheduled for 28 November 2026.

The final review date and review designation should be confirmed against the latest communication from the department.

---

# 2. Project Overview

TB-Sentry is an explainable deep learning-based research prototype for tuberculosis screening from chest X-ray images.

The system uses DenseNet-121 for binary TB-labelled versus non-TB-labelled classification.

Grad-CAM is used to generate visual explanations of model predictions. These explanations are compared against available TB localization bounding boxes to investigate spatial consistency.

The project additionally explores whether model confidence is associated with explanation consistency.

The project is intended for academic research and demonstration. It has not been clinically validated.

---

# 3. Problem Definition

Deep learning models can classify medical images, but their predictions are not always straightforward to interpret.

A classification score alone does not reveal which image regions influenced the prediction.

TB-Sentry investigates whether visual explanations and quantitative spatial consistency measurements can provide additional information about model behaviour.

The central research question is:

**Can the spatial consistency of Grad-CAM explanations with available TB localization annotations provide useful information alongside model confidence when examining potentially unreliable predictions?**

This question remains exploratory. The current experiments do not establish that explanation consistency improves human-review decisions.

---

# 4. Project Objectives

1. Build a reproducible chest X-ray classification pipeline.
2. Implement DenseNet-121-based binary classification.
3. Evaluate performance using appropriate classification metrics.
4. Investigate the effect of exact-duplicate removal.
5. Evaluate variation across multiple random seeds.
6. Generate Grad-CAM explanations.
7. Measure spatial consistency with localization bounding boxes.
8. Investigate confidence-localization associations.
9. Implement a demonstration interface.
10. Document experimental findings and limitations.

---

# 5. Implementation Completed

| Module | Implementation Status |
|---|---|
| Dataset ingestion | Completed |
| Manifest generation | Completed |
| Dataset validation | Completed |
| Image preprocessing | Completed |
| DenseNet-121 baseline | Completed |
| Threshold analysis | Completed |
| Exact-duplicate audit | Completed |
| Duplicate-aware classification | Completed |
| Targeted near-duplicate exclusion | Completed |
| Multiseed evaluation | Completed |
| Grad-CAM | Completed |
| Spatial consistency evaluation | Completed |
| Confidence association analysis | Completed |
| Automated testing | Completed |
| Streamlit demonstration | Implemented and run |
| Documentation | Updated |

---

# 6. Dataset Summary

The primary dataset is TBX11K.

| Split | Total Images | TB-labelled | Non-TB-labelled |
|---|---:|---:|---:|
| Training | 6,600 | 600 | 6,000 |
| Validation | 1,800 | 200 | 1,600 |
| **Total** | **8,400** | **800** | **7,600** |

The binary mapping is:

- `health` → Non-TB-labelled (0)
- `sick` → Non-TB-labelled (0)
- `tb` → TB-labelled (1)

The validation set contains 200 images with localization annotations and 309 bounding boxes.

The downloaded test-set discrepancy remains unresolved. Therefore, the reported results are validation results, not independent test results.

---

# 7. Experimental Results

## 7.1 Classification Comparison

| Metric | Baseline | Duplicate-Aware | Near-Duplicate-Excluded |
|---|---:|---:|---:|
| Sensitivity | 49.50% | 72.00% | 69.00% |
| Specificity | 99.75% | 97.75% | 98.50% |
| Precision | 96.12% | 80.00% | 85.19% |
| F1-score | 0.6535 | 0.7579 | 0.7624 |
| ROC-AUC | 0.95985 | 0.95444 | 0.95393 |
| Average precision | 0.83716 | 0.84338 | 0.84175 |

All three experiments use threshold 0.50 and the same validation set.

## 7.2 Multiseed Evaluation

Three seeds were evaluated:

- Seed 42
- Seed 123
- Seed 2026

Mean results:

| Metric | Mean |
|---|---:|
| Sensitivity | 78.67% |
| Specificity | 95.48% |
| Precision | 69.74% |
| F1-score | 0.7345 |
| ROC-AUC | 0.9615 |
| Average precision | 0.8507 |

## 7.3 Grad-CAM Spatial Consistency

At heatmap threshold 0.50:

| Metric | Mean | Median |
|---|---:|---:|
| Union box coverage | 20.21% | 6.94% |
| Heatmap precision | 9.88% | 2.62% |
| Mean individual box coverage | 19.78% | 6.74% |

The measurements indicate that spatial overlap varies across images and depends on the heatmap threshold.

## 7.4 Confidence Association

| Spatial Metric | Spearman Correlation |
|---|---:|
| Union box coverage | 0.3014 |
| Heatmap precision | 0.1712 |
| Mean individual box coverage | 0.2912 |

The observed correlations are positive. They do not establish causality or prove that explanation consistency improves classification or review decisions.

---

# 8. Software Verification

The automated test suite was executed using:

```cmd
set PYTHONPATH=src
.venv_test\Scripts\python.exe -m pytest -q
```

Recorded result:

| Test Result | Count |
|---|---:|
| Passed | 13 |
| Failed | 0 |

The Streamlit demonstration interface was also run successfully.

The demonstration is a research interface and must not be presented as a clinically validated diagnostic tool.

---

# 9. Demonstration Plan

The review demonstration should follow this sequence:

1. Introduce the TB-Sentry problem statement.
2. Explain the TBX11K dataset and binary label mapping.
3. Show the dataset preparation and duplicate-audit process.
4. Explain the DenseNet-121 architecture and preprocessing.
5. Demonstrate the classification interface.
6. Show a sample chest X-ray classification workflow.
7. Explain the prediction score and its limitations.
8. Display the corresponding Grad-CAM heatmap.
9. Explain spatial consistency using bounding-box annotations.
10. Present the experimental results.
11. Explain limitations and future work.

Only use actual project outputs and verified results during the demonstration.

Do not describe a heatmap as a confirmed lesion segmentation.

---

# 10. University Review Evaluation Rubric

The supplied university notice specifies the following 50-mark evaluation structure:

| Criterion | Description | Marks |
|---|---|---:|
| Problem Definition & Literature Survey | Problem statement, objectives, literature survey, research gap, and relevance to SIH/SDG | 10 |
| Methodology & Technical Design | Methodology, architecture, workflow, technology stack, model selection, and feasibility | 10 |
| Implementation Progress | Actual implementation, working modules, prototype, results, screenshots, and technical progress | 10 |
| Presentation & Team Contribution | Presentation quality, individual contribution, teamwork, challenges, and future work | 10 |
| Patent Publication | Patent publication | 10 |
| **Total** | | **50** |

The patent publication requirement should be discussed with the project guide or department. This document does not claim that a patent has been filed, published, or granted.

---

# 11. Questions to Prepare For

The following questions should be rehearsed before the review.

## Problem and Research Gap

1. Why is tuberculosis screening from chest X-rays an important research problem?
2. What limitations of ordinary image classification does TB-Sentry investigate?
3. What is the difference between explainability and model accuracy?
4. What is the central research question?

## Dataset

5. Why was TBX11K selected?
6. How many images are present in the training and validation splits?
7. How were the original categories mapped into binary labels?
8. What is the significance of the class imbalance?
9. What are localization bounding boxes?
10. Why are bounding boxes not equivalent to segmentation masks?

## Model and Methodology

11. Why was DenseNet-121 selected?
12. Why were ImageNet-pretrained weights used?
13. Why was the backbone frozen?
14. What is weighted cross-entropy?
15. What is the purpose of threshold analysis?
16. What are sensitivity and specificity?

## Duplicate Auditing

17. Why is duplicate removal important?
18. How does SHA-256 identify exact duplicate images?
19. What is the difference between exact duplicates and near-duplicates?
20. Does exact duplicate removal establish patient-level independence?

## Explainability

21. What is Grad-CAM?
22. How is a Grad-CAM heatmap generated?
23. How is spatial consistency measured?
24. Why is heatmap thresholding necessary?
25. Does a high spatial overlap prove that the explanation is clinically correct?

## Experimental Results

26. Why did sensitivity change between experiments?
27. Why can specificity decrease when sensitivity increases?
28. Why were three random seeds used?
29. What does ROC-AUC measure?
30. What does average precision measure?
31. Why is the validation set not an independent final test set?

## Limitations and Future Work

32. Why is one epoch insufficient to establish convergence?
33. Why is patient-level validation important?
34. What is probability calibration?
35. What additional validation would be required before clinical use?
36. What would be the next technical improvements?

---

# 12. Known Limitations to Communicate

The following points should be acknowledged clearly:

1. The classification experiments used one training epoch.
2. The backbone remained frozen.
3. The multiseed experiment includes only three seeds.
4. The validation set was reused across multiple experiments.
5. Patient-level independence has not been established.
6. The downloaded test-set discrepancy remains unresolved.
7. Exact duplicate removal does not eliminate every possible near-duplicate.
8. Bounding boxes provide coarse localization annotations.
9. Grad-CAM is a post-hoc explanation technique.
10. Model confidence has not been calibrated.
11. Clinical validation has not been performed.
12. The system is not suitable for independent medical diagnosis.

---

# 13. Future Work

Potential future improvements include:

- Longer training schedules.
- Additional random seeds.
- Patient-level split verification.
- Independent test-set evaluation.
- Probability calibration.
- More rigorous explanation evaluation.
- Further investigation of confidence and spatial consistency.
- External validation after establishing dataset independence.
- Clinical collaboration and evaluation.

---

# 14. Meeting Preparation Checklist

Before the review:

- [ ] Verify the project repository.
- [ ] Confirm the demonstration environment.
- [ ] Confirm the required model checkpoint is available locally.
- [ ] Prepare representative chest X-ray examples.
- [ ] Prepare Grad-CAM visualizations.
- [ ] Prepare classification result tables.
- [ ] Prepare duplicate-audit evidence.
- [ ] Prepare spatial consistency results.
- [ ] Review the methodology document.
- [ ] Review the experiment log.
- [ ] Review the project plan.
- [ ] Prepare an explanation of individual project contributions.
- [ ] Prepare answers to technical questions.
- [ ] Confirm the review date and presentation requirements with the department.
- [ ] Clarify the patent publication requirement with the project guide.

---

# 15. Final Review Status

TB-Sentry has progressed from dataset preparation and baseline classification to duplicate-aware experimentation, multiseed evaluation, Grad-CAM visualization, spatial consistency analysis, confidence-association analysis, software testing, and demonstration.

The project currently has a functioning research prototype and documented preliminary experimental results.

The review presentation should emphasize the actual implementation, experimental evidence, research question, technical challenges, and limitations.

**Current status:** Core prototype implemented and experimentally evaluated. Review preparation in progress.