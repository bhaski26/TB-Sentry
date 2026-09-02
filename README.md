# TB-Sentry

### Explainable AI for Tuberculosis Screening from Chest X-Rays

**Academic Project Title:**
**TUBERCULOSIS SCREENING FROM CHEST X-RAYS WITH EXPLAINABILITY**

TB-Sentry is an academic research prototype that investigates deep learning–based tuberculosis screening from chest X-ray images with interpretable model predictions.

The system is designed to explore not only whether a chest X-ray is classified as suggestive of tuberculosis, but also **where the model focuses when making its prediction** and whether the generated explanation is spatially consistent with available tuberculosis-related localization annotations.

## Proposed Pipeline

```text
Chest X-ray
    │
    ▼
Preprocessing
    │
    ▼
Lung Segmentation
    │
    ▼
TB Classification
    │
    ├──────────────► TB Probability
    │
    ▼
Grad-CAM Explanation
    │
    ▼
Spatial Consistency Evaluation
    │
    ▼
Confidence + Explanation Consistency
    │
    ▼
Prediction / Human Review Signal
```

## Proposed Models

| Component              | Proposed Approach                               |
| ---------------------- | ----------------------------------------------- |
| Preprocessing          | CLAHE + normalization                           |
| Lung segmentation      | U-Net                                           |
| TB classification      | DenseNet-121                                    |
| Explainability         | Grad-CAM                                        |
| Explanation evaluation | IoU / Dice                                      |
| Reliability analysis   | Prediction confidence + explanation consistency |

> **Note:** These are the initial proposed components and will be validated experimentally during development. Alternative architectures may be evaluated if justified by the experiments and available computational resources.

## Datasets

The project will investigate publicly available chest X-ray datasets including:

* Shenzhen Hospital Chest X-ray Dataset
* Montgomery County Chest X-ray Dataset
* TBX11K

Additional datasets may be considered for pretraining or external evaluation if they are methodologically appropriate.

**Datasets will not be stored directly in this repository.**

## Research Focus

The primary research focus is the investigation of **explainable and reliability-aware tuberculosis screening**.

Rather than treating a Grad-CAM heatmap as proof that a prediction is trustworthy, TB-Sentry will investigate whether explanation localization is spatially consistent with available reference annotations and whether this information can complement prediction confidence when identifying potentially unreliable cases.

## Project Status

🚧 **Active Research / Development**

This repository contains an academic research prototype and is not intended for clinical diagnosis or to replace qualified medical professionals.

## Team

**TB-Sentry — Final-Year B.Tech CSE Capstone Project**

Team members: *To be added*

## License

License information will be added once the project's code-sharing and publication requirements are finalized.
