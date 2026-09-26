# Step #3 — Hybrid OpenCV + PyTorch Inspection Design

**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-25

## Objective
Step #3 converts the Step #2 synthetic acquisition layer into an executable inspection pipeline. Classical CV performs part localization, pose estimation, rectification, and geometry integrity analysis. A compact PyTorch U-Net performs binary scratch segmentation. This separation is intentional: deterministic geometry has strong priors, while scratch appearance benefits from learned segmentation.

## Processing flow
1. Normalize and lightly denoise the grayscale image.
2. Threshold the bright part and retain the largest external contour.
3. Estimate center, orientation, and dimensions with `minAreaRect`.
4. Rectify the part to a canonical ROI.
5. Measure contour rectangularity, solidity, and polygon corner count for geometry damage.
6. Train a compact U-Net on **normal + scratch** samples only.
7. Infer a scratch probability map and threshold it into a binary segmentation mask.
8. Fuse geometry and scratch decisions into PASS/REJECT.
9. Save preprocessing, localization, rectification, probability, mask, JSON result, model, and training metrics.

## Important label limitation
Step #2 has one combined defect mask. For `mixed` samples that mask does not distinguish scratch pixels from chip pixels. Therefore Step #3 deliberately excludes `mixed`, `edge_damage`, and `corner_damage` from U-Net training rather than introducing label leakage. Geometry defects remain the responsibility of deterministic CV. A future dataset version should export per-defect masks (`scratch_mask`, `edge_mask`, `corner_mask`) to permit multi-task training.

## Quantitative validation
The U-Net records train/validation loss and pixel precision, recall, Dice/F1, and IoU. The test split is evaluated only after validation-based model selection. Classical CV tests verify localization and geometric plausibility. Step #4 will add full dataset-level part decision metrics, latency distributions, confusion matrices, and PLC/system integration.

## Production caveat
All Step #3 metrics are synthetic-domain software feasibility evidence. They are not factory acceptance results. Real optical data, MSA/calibration, domain-gap validation, threshold tuning on representative validation data, and independent acceptance testing remain required.
