# \# TB-Sentry — Implementation Progress Log

#

# \*\*Project title:\*\* TUBERCULOSIS SCREENING FROM CHEST X-RAYS WITH EXPLAINABILITY

# \*\*Project name:\*\* TB-Sentry

# \*\*Status recorded:\*\* 12 October 2026

#

# \## 1. Project Objective

#

# TB-Sentry investigates a chest-X-ray tuberculosis screening classifier and whether quantitative spatial consistency of model explanations with available localization annotations can provide useful information for flagging predictions for human review.

#

# The intended pipeline is:

#

# 1\. Prepare and validate dataset manifests.

# 2\. Train a binary TB/non-TB classifier.

# 3\. Evaluate classification performance with class-imbalance-aware metrics.

# 4\. Generate Grad-CAM explanations.

# 5\. Compare heatmap locations with available TB localization annotations.

# 6\. Investigate whether explanation consistency complements model confidence when identifying predictions for review.

# 7\. Build a small demonstration interface if time permits.

#

# The project is an academic research prototype, not a clinically validated diagnostic system.

#

# \## 2. Current Scope and Terminology

#

# The initial binary mapping is:

#

# \- `health` → non-TB-labelled (0)

# \- `sick` → non-TB-labelled (0)

# \- `tb` → TB-labelled (1)

#

# This combines the available TB-labelled categories for the first experiment. It must not be described as a classifier specifically validated for active TB, nor as evidence of clinical generalization.

#

# TBX11K bounding boxes are localization annotations, not pixel-level lesion masks. Any heatmap-to-box measurement should be described as \*\*spatial consistency with localization annotations\*\*, not proof that the explanation identifies the true clinical lesion.

#

# The central research question remains an investigation: does explanation consistency provide useful information beyond classifier confidence? Improvement has not yet been demonstrated.

#

# \## 3. Dataset Preparation and Audit

#

# The downloaded TBX11K package is stored locally under:

#

# `data/raw/tbx11k/TBX11K/`

#

# Dataset images and annotations are excluded from version control. Refer to `data/README.md` for data provenance and handling rules.

#

# A manifest-building script, `scripts/build\_manifest.py`, was implemented. It reads the official TBX11K train/validation lists and corresponding JSON annotation records, maps source folders to binary labels, and writes the ignored intermediate manifest:

#

# `data/interim/tbx11k\_manifest.csv`

#

# \### Manifest Schema

#

# The manifest contains:

#

# \- `image\_path`

# \- `split`

# \- `source\_category`

# \- `binary\_label`

# \- `image\_width`

# \- `image\_height`

# \- `localization\_boxes`

# \- `localization\_categories`

# \- `annotation\_status`

#

# \### Manifest Audit Results

#

# | Split | Total | TB-labelled | Non-TB-labelled | Images with boxes | Images without boxes |

# |---|---:|---:|---:|---:|---:|

# | Train | 6,600 | 600 | 6,000 | 599 | 6,001 |

# | Validation | 1,800 | 200 | 1,600 | 200 | 1,600 |

# | Total | 8,400 | 800 | 7,600 | 799 | 7,601 |

#

# \### Validation Checks

#

# The following checks passed:

#

# \- Every listed image exists.

# \- Official list entries match the corresponding JSON image records.

# \- No duplicate image paths occur across the official train and validation lists.

# \- Every image has a recognized source category.

# \- Binary labels follow the configured mapping.

#

# \*\*Important limitation:\*\* path-level split checks do not establish patient-level independence or rule out duplicate image content across datasets. These checks also do not establish clinical validity.

#

# The training set has a 1:10 TB-labelled-to-non-TB-labelled ratio. The validation set has a 1:8 ratio.

#

# This imbalance motivates reporting sensitivity, specificity, precision, F1-score, ROC-AUC, average precision, and confusion matrices rather than relying on accuracy alone.

#

# \### Localization Annotation Availability

#

# The training JSON contains 6,600 image records and 902 bounding boxes across 599 images.

#

# The validation JSON contains 1,800 image records, of which 200 have bounding boxes.

#

# Most images do not have localization boxes. Spatial-consistency analyses must therefore use only the subset with suitable annotations and explicitly report the number of evaluated images.

#

# \### Unresolved Test-Set Discrepancy

#

# The downloaded package contains 3,302 files in its `test` directory, while published split information reports a 2,800-image test split. The local `all\_test.json` contains no bounding boxes.

#

# This discrepancy remains unresolved.

#

# The downloaded test directory must not be treated as an independent labelled test set without further verification.

#

# \## 4. Data Loading Implementation

#

# Implemented components:

#

# \- `src/tb\_sentry/data/dataset.py`

# \- `src/tb\_sentry/data/transforms.py`

# \- `tests/test\_dataset.py`

# \- `scripts/test\_dataloader.py`

#

# The dataset class:

#

# \- Reads the CSV manifest using Python's built-in `csv` module.

# \- Loads images using Pillow.

# \- Resizes images to 224 × 224.

# \- Converts images to tensors.

# \- Applies ImageNet normalization.

# \- Returns image tensors, labels, image paths, source categories, localization boxes, localization categories, and annotation status.

#

# \### Unit Test Results

#

# Five dataset tests passed in `.venv\_test`:

#

# 1\. Training dataset counts.

# 2\. Validation dataset counts.

# 3\. Training sample format.

# 4\. Validation sample format.

# 5\. Rejection of an unsupported test split.

#

# \### DataLoader and Model Smoke Test

#

# The smoke test passed with:

#

# \- Image batch shape: `(4, 3, 224, 224)`

# \- Label batch shape: `(4,)`

# \- Image dtype: `torch.float32`

# \- Model output shape: `(4, 2)`

#

# The test also confirmed that variable-length localization metadata could be preserved by the custom collate function.

#

# \### Environment Notes

#

# Development is being performed on Windows with Python 3.11 and CPU-only PyTorch.

#

# Torchvision emits a warning because its native image extension is blocked by a Windows application-control policy. Consequently, `torchvision.io` image reading is not used. Pillow image loading works.

#

# The original environment's pandas import failed because a compiled DLL was blocked. Current scripts use the standard-library CSV reader rather than depending on pandas.

#

# The Windows security policy must not be bypassed or disabled to accommodate the project.

#

# \## 5. Baseline Classification Model

#

# Implemented components:

#

# \- `src/tb\_sentry/classification/densenet.py`

# \- `scripts/train\_baseline.py`

# \- `scripts/export\_validation\_predictions.py`

#

# \### Model Configuration

#

# | Parameter | Value |

# |---|---|

# | Architecture | DenseNet-121 |

# | Initialization | ImageNet-pretrained weights |

# | Backbone | Frozen |

# | Classification head | New two-class linear layer |

# | Total parameters | 6,955,906 |

# | Trainable parameters | 2,050 |

# | Optimizer | Adam |

# | Learning rate | 0.001 |

# | Batch size | 4 |

# | Epochs | 1 |

# | Random seed | 42 |

# | Loss function | Class-weighted cross-entropy |

# | Device | CPU |

# | Initial classification threshold | 0.50 |

#

# The class weights for `\[non-TB, TB]` were approximately:

#

# `\[0.181818, 1.818182]`

#

# The weights compensate for the imbalance in the training labels.

#

# \### Generated Artifacts

#

# The experiment produced:

#

# \- `models/checkpoints/densenet121\_baseline.pt`

# \- `results/baseline\_metrics.json`

# \- `results/baseline\_metrics.csv`

# \- `results/tables/baseline\_validation\_predictions.csv`

#

# The checkpoint is a local artifact and must not be committed to Git.

#

# \### Initial Validation Results

#

# | Metric | Result |

# |---|---:|

# | Validation loss | 0.15485 |

# | Sensitivity | 49.50% |

# | Specificity | 99.75% |

# | Precision | 96.12% |

# | F1-score | 0.6535 |

# | ROC-AUC | 0.9599 |

# | Average precision | 0.8372 |

# | True negatives | 1,596 |

# | False positives | 4 |

# | False negatives | 101 |

# | True positives | 99 |

#

# These results come from one epoch, one seed, and the current validation split. They are preliminary, not a final performance estimate.

#

# At threshold 0.50, the model misses 101 of the 200 TB-labelled validation images. This is an important limitation for a screening-oriented project.

#

# \*\*Metric naming note:\*\* the current training script's `pr\_auc` field stores scikit-learn average precision. In formal reporting, call this \*\*average precision (AP)\*\* unless trapezoidal area under the precision-recall curve is explicitly calculated.

#

# \## 6. Threshold Analysis

#

# The script `scripts/analyze\_thresholds.py` evaluates thresholds from 0.05 to 0.95 in increments of 0.05 using the saved validation predictions.

#

# It produces:

#

# \- `results/tables/baseline\_threshold\_analysis.csv`

# \- `results/figures/baseline\_threshold\_analysis.png`

#

# \### Selected Results

#

# | Threshold | Sensitivity | Specificity | Precision | F1-score | Balanced Accuracy | FN | FP |

# |---|---:|---:|---:|---:|---:|---:|---:|

# | 0.10 | 0.815 | 0.938 | 0.622 | 0.706 | 0.877 | 37 | 99 |

# | 0.20 | 0.735 | 0.979 | 0.812 | 0.772 | 0.857 | 53 | 34 |

# | 0.30 | 0.655 | 0.989 | 0.885 | 0.753 | 0.822 | 69 | 17 |

# | 0.50 | 0.495 | 0.998 | 0.961 | 0.653 | 0.746 | 101 | 4 |

#

# Among the evaluated thresholds, 0.20 produced the highest F1-score:

#

# \- Sensitivity: 73.5%

# \- Specificity: 97.9%

# \- Precision: 81.2%

# \- F1-score: 0.772

# \- False negatives: 53

# \- False positives: 34

#

# \### Interpretation

#

# Lowering the threshold increases sensitivity but also increases false positives.

#

# At threshold 0.10, the model detects more TB-labelled validation images but produces substantially more false positives.

#

# At threshold 0.30, the model produces fewer false positives but misses more TB-labelled validation images.

#

# Threshold 0.20 provides the highest F1-score among the thresholds examined. However, it is only a descriptive result on the current validation set.

#

# Selecting a threshold on the validation set and reporting performance on that same set introduces selection optimism. Therefore, threshold 0.20 must not be presented as a clinically appropriate operating point or an unbiased final performance estimate.

#

# The model's output scores have not yet been shown to be calibrated probabilities. Calibration should be evaluated before confidence is used in the proposed trust layer.

#

# \## 7. Reproducibility and Version-Control Policy

#

# The repository should contain source code, tests, configuration, documentation, and selected small result summaries.

#

# Do not commit:

#

# \- Raw datasets or annotations.

# \- Generated manifests under `data/interim/`.

# \- Model checkpoint files.

# \- Large generated artifacts or caches.

# \- Secrets or machine-specific absolute paths.

#

# Verify `.gitignore` before staging files.

#

# \## 8. Known Limitations and Open Risks

#

# 1\. Only one training epoch and one random seed have been evaluated.

# 2\. The backbone is frozen; alternative training configurations have not been compared.

# 3\. Patient-level independence and content-level duplicates have not been established.

# 4\. The test directory/list discrepancy remains unresolved.

# 5\. External dataset overlap has not been ruled out.

# 6\. Most images do not have localization boxes.

# 7\. Bounding boxes are coarse localization annotations, not pixel-level ground truth.

# 8\. Confidence calibration has not been assessed.

# 9\. No Grad-CAM results or explanation-consistency results have been produced yet.

# 10\. No evidence yet shows that explanation consistency improves review decisions over confidence alone.

#

# \## 9. Next Steps

#

# Priority order:

#

# 1\. Commit the current validated code and documentation without raw data or checkpoints.

# 2\. Implement Grad-CAM for the trained DenseNet-121.

# 3\. Save a small set of qualitative explanation examples.

# 4\. Add tests for heatmap dimensions, finite values, and inference consistency.

# 5\. Define a fixed heatmap normalization and thresholding protocol before measuring spatial consistency.

# 6\. Compare heatmaps against available localization boxes on the annotated subset only.

# 7\. Report box-based spatial consistency without claiming pixel-level clinical correctness.

# 8\. Evaluate whether spatial consistency adds information beyond confidence.

# 9\. Revisit training duration and repeatability after the explanation pipeline works.

# 10\. Resolve test-data provenance and investigate patient/content-level leakage before making generalization claims.

# 11\. Add a demonstration interface only after the core experiment is reproducible.

#

# \## 10. Review Readiness

#

# For the first project review, the current evidence supports demonstrating:

#

# \- Dataset audit and manifest generation.

# \- Tested data loading.

# \- A working DenseNet-121 baseline.

# \- Validation metrics.

# \- Threshold trade-offs.

#

# Explainability and trust-layer experiments remain pending and must be presented as planned work, not completed results.
