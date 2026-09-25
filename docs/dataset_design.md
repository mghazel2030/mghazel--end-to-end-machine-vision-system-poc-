# Step 2 — Synthetic dataset engineering

## Goal
Provide reproducible, sufficiently varied labeled data for Step 3 algorithm
development because no customer dataset was supplied.

## Domain randomization
Each sample independently varies surface gray level, low-frequency matte
texture, illumination gradient, Gaussian sensor-like noise, blur, pose,
translation and legitimate alphanumeric markings. These nuisance variables
are deliberately **not** defect labels.

## Exact labels
- `normal`: zero mask and no bounding box.
- `scratch`: dark line-like defect with exact pixel mask.
- `edge_damage`: material removal at a nominal straight edge.
- `corner_damage`: material removal at a nominal corner.
- `mixed`: scratch plus geometric damage.
Bounding boxes are derived from nonzero masks after pose transformation.

## Splitting
The committed default is 400 images for a fast reproducible PoC run and
70/15/15 train/validation/test partitions. The generator is deterministic by
seed. Before Step 3 final training, increase `total_images` (for example 4,000)
after runtime/storage verification. The held-out test partition must not be
used for threshold tuning.

## Important limitation
Synthetic data is an engineering-development surrogate. It does not reproduce
the complete BRDF, lens MTF/distortion, vibration, contamination, strobe
nonuniformity, sensor response or manufacturing defect distribution. Real
factory data must be collected and independently evaluated before FAT/SAT.
