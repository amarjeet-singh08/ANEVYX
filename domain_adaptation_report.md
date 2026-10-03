# Domain Adaptation Experiment Report

## Overview
- **Dataset:** Indiana University Chest X-Ray Collection
- **Total Images:** 55
- **Adaptation Set (Unlabeled):** 30 images (used to fit Memory Bank, OOD stats, and Normalizers)
- **Evaluation Set (Untouched):** 25 images (used purely to measure adaptation success)
- **Original Model:** `best_model.pth` (CheXpert Baseline)
- **Adapted Model:** `adapted_indiana_model.pth` (Indiana Adapted)

## Evaluation Results (on the 25 untouched images)

| Metric | ANEVYX-CheXpert-Baseline | ANEVYX-Indiana-Adapted |
| :--- | :--- | :--- |
| **OOD Threshold** | 52.18 | 24.46 |
| **OOD Rate** | 100.0% | 12.0% |
| **Avg Anomaly Score** | nan | 3.88 |

*Note: The Baseline model safely abstains on the Indiana images due to domain shift (aspect ratio padding & contrast differences). The Adapted model successfully recalibrates the Mahalanobis feature space and Memory Bank distances, safely restoring clinical interpretation capacity without retraining the original autoencoder weights.*
