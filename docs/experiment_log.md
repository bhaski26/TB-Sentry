# \# TB-Sentry — Experiment Log

#

# \## EXP-000: TBX11K Manifest Audit

#

# \*\*Status:\*\* Completed

# \*\*Purpose:\*\* Build a reproducible train/validation index without copying dataset images into Git.

#

# \### Dataset Summary

#

# | Split | Total Images | TB-labelled | Non-TB-labelled |

# |---|---:|---:|---:|

# | Train | 6,600 | 600 | 6,000 |

# | Validation | 1,800 | 200 | 1,600 |

# | Total | 8,400 | 800 | 7,600 |

#

# \### Localization Annotations

#

# \- Training images with boxes: 599.

# \- Bounding boxes in the training JSON: 902.

# \- Validation images with boxes: 200.

# \- Most images have no localization boxes.

#

# \### Checks Performed

#

# \- Every listed image path exists.

# \- List records match the corresponding JSON records.

# \- No duplicate image paths occur across train and validation lists.

# \- Source categories are recognized.

# \- Binary labels follow the configured mapping.

#

# These checks validate paths and metadata. They do not establish patient-level independence, clinical validity, or absence of duplicate image content.

#

# \### Unresolved Test-Set Issue

#

# The downloaded package contains 3,302 test images, while published split information reports 2,800. The local `all\_test.json` contains no bounding boxes.

#

# This discrepancy remains unresolved. Do not use the downloaded directory as a labelled independent test set without further verification.

#

# \---

#

# \## EXP-001: Frozen DenseNet-121 Baseline

#

# \*\*Status:\*\* Completed; preliminary

# \*\*Task:\*\* Binary TB-labelled versus non-TB-labelled classification

# \*\*Dataset:\*\* TBX11K official train and validation splits as represented in `data/interim/tbx11k\_manifest.csv`

# \*\*Date recorded:\*\* 12 October 2026

#

# \### Configuration

#

# | Parameter | Value |

# |---|---|

# | Architecture | DenseNet-121 |

# | Pretrained weights | ImageNet |

# | Backbone | Frozen |

# | Classification head | Two-class linear layer |

# | Training examples | 6,600 |

# | Validation examples | 1,800 |

# | Training positives | 600 |

# | Training negatives | 6,000 |

# | Validation positives | 200 |

# | Validation negatives | 1,600 |

# | Batch size | 4 |

# | Epochs | 1 |

# | Optimizer | Adam |

# | Learning rate | 0.001 |

# | Seed | 42 |

# | Loss | Weighted cross-entropy |

# | Class weights `\[non-TB, TB]` | `\[0.181818, 1.818182]` |

# | Device | CPU |

# | Initial threshold | 0.50 |

#

# \### Results at Threshold 0.50

#

# | Metric | Result |

# |---|---:|

# | Validation loss | 0.1548466 |

# | Sensitivity | 0.4950 |

# | Specificity | 0.9975 |

# | Precision | 0.9612 |

# | F1-score | 0.6535 |

# | ROC-AUC | 0.95985 |

# | Average precision | 0.83716 |

# | True negatives | 1,596 |

# | False positives | 4 |

# | False negatives | 101 |

# | True positives | 99 |

#

# \### Threshold Sweep

#

# Thresholds from 0.05 to 0.95, in increments of 0.05, were evaluated using the same validation predictions.

#

# The best F1-score among the evaluated thresholds was 0.772 at threshold 0.20:

#

# \- Sensitivity: 0.735.

# \- Specificity: 0.979.

# \- Precision: 0.812.

# \- False negatives: 53.

# \- False positives: 34.

#

# \### Interpretation

#

# The model has a strong preliminary ROC-AUC, but at threshold 0.50 it misses approximately half of the TB-labelled validation cases.

#

# The threshold sweep illustrates the trade-off between sensitivity and specificity.

#

# Threshold 0.20 is not declared a clinical operating threshold. It was selected using the current validation set, so reporting the same validation results as an unbiased final estimate would be inappropriate.

#

# The output scores have not yet been calibrated.

#

# \### Artifacts

#

# \- `models/checkpoints/densenet121\_baseline.pt` — local checkpoint; do not commit.

# \- `results/baseline\_metrics.json`

# \- `results/baseline\_metrics.csv`

# \- `results/tables/baseline\_validation\_predictions.csv`

# \- `results/tables/baseline\_threshold\_analysis.csv`

# \- `results/figures/baseline\_threshold\_analysis.png`

#

# \### Limitations

#

# 1\. Single epoch.

# 2\. Single seed.

# 3\. Frozen backbone.

# 4\. No independent final test evaluation.

# 5\. Patient-level and content-level leakage checks are incomplete.

# 6\. Probability calibration has not been assessed.

# 7\. Grad-CAM and spatial-consistency experiments are pending.

#

# \---

#

# \## Next Experiment

#

# \*\*EXP-002: Grad-CAM Explanation Generation\*\*

#

# Planned objectives:

#

# 1\. Load the saved DenseNet-121 checkpoint.

# 2\. Generate Grad-CAM heatmaps for selected validation images.

# 3\. Overlay heatmaps on the original chest X-rays.

# 4\. Verify that heatmaps have valid dimensions and finite values.

# 5\. Compare examples across true positives, false positives, true negatives, and false negatives.

# 6\. Save reproducible visual outputs without committing large datasets or checkpoints.

#

# Following qualitative verification, implement a predefined box-based spatial-consistency metric for images with localization annotations.

#

# The metric must be described as spatial consistency with coarse localization annotations, not proof of clinical correctness.
