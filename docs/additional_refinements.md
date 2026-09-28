# Additional Presentation and Quantitative-Evaluation Refinements

**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-27

## Purpose

This refinement adds presentation-ready intermediate images and quantitative evidence without changing the core engineering architecture. All artifacts are generated from the same deterministic dataset and held-out test split used by the executable pipeline.

## Common-image processing sequence

`results/inspection_run/10_presentation_evidence/01_processing_sequence/` contains one common held-out image followed through the exact sequence:

```text
Common test image
      ↓
Gaussian denoise + intensity normalization
      ↓
Foreground threshold + morphology
      ↓
Largest contour
      ↓
Minimum-area rotated rectangle
      ↓
Centroid + orientation estimate
      ↓
Rotation / canonical crop
      ↓
Rectified ROI
```

The folder contains the original, preprocessed, localized, orientation-annotated, and rectified images so the effect of each operation can be shown side-by-side.

## Conventional Computer Vision geometry pipeline

The implemented CCV geometry branch is primarily **part-level defect detection**, not a learned defect-segmentation model:

```text
Preprocessed image
      ↓
Foreground segmentation
      ↓
Morphological closing
      ↓
Largest part contour
      ↓
Minimum-area rectangle
      ↓
┌───────────────────────────────┐
│ Rectangularity = contour/box  │
│ Solidity = contour/hull       │
│ Polygon corner count          │
└───────────────────────────────┘
      ↓
Threshold rules
      ↓
Geometry PASS / REJECT
```

An additional **geometry discrepancy mask** is generated for visualization by subtracting the observed part foreground from its ideal fitted rotated rectangle. This provides approximate localization/segmentation of missing material, but the production decision remains based on interpretable part-level geometry features.

### CCV evaluation scope

- **Part-level metrics:** all held-out test images. Geometry-positive ground truth is `edge_damage`, `corner_damage`, or `mixed`; `normal` and `scratch` are geometry-negative.
- **Pixel-level discrepancy metrics:** only `normal`, `edge_damage`, and `corner_damage`, because these have unambiguous geometry masks. `mixed` is excluded from pixel metrics because its single synthetic mask combines scratch and geometry pixels.
- **Overlays:** generated for every held-out test image. Yellow is synthetic ground truth; red is the CCV discrepancy estimate.

## U-Net evaluation scope

The scratch U-Net is evaluated on **all eligible held-out normal and scratch images**. It is not scored as a scratch segmenter on edge/corner/mixed images because the current synthetic labels do not provide class-separated scratch masks for mixed defects.

Artifacts include:

- train/validation/test counts for the full dataset and U-Net-eligible subset;
- one representative image from each split;
- architecture diagram;
- loss and Dice learning curves;
- per-image test metrics;
- aggregate Precision, Recall, Dice/F1, and IoU;
- ground-truth/prediction overlays for every eligible U-Net test image.

## Decision-fusion evaluation scope

The calibrated hybrid CCV + U-Net decision is evaluated on **every held-out test image**. The validation split is used to select the scratch-fraction threshold; the test split remains protected until final evaluation. Part-level Accuracy, Precision, Recall, Specificity, F1, False-Accept Rate, and False-Reject Rate are reported and plotted.

## Dataset size

The default synthetic dataset is increased from 400 to **600 images**. With a 70/15/15 split this provides 420 training, 90 validation, and 90 held-out test images overall. Because the U-Net intentionally uses only class-pure `normal` and `scratch` samples, its effective subset is smaller. Six hundred images is a practical CPU-only compromise: it increases the evaluation sample count by 50% while keeping generation and three-epoch compact-U-Net training reasonable on a laptop without a GPU.

For final scientific characterization, use repeated seeds and substantially more real factory data rather than treating a larger synthetic dataset as a substitute for real-domain validation.
