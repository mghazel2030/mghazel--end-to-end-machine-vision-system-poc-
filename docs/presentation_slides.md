# Copy/Paste Presentation Slides — Cumulative through Step #3

## Slide 1 — End-to-End Machine-Vision PoC
- Conveyor inspection of matte rectangular manufactured parts.
- Requirements/hardware → synthetic acquisition → CV/AI → integration → production readiness.
- Current implementation: Steps #1–#3.

## Slide 2 — Application & Defects
- 200×120 mm part; ±20° orientation; 0.5 m/s conveyor.
- Scratch/cut: ≥0.5 mm wide, ≥5 mm long.
- Edge/corner damage: ≥2 mm.
- Legitimate text/logos and nuisance variation included.

## Slide 3 — Quantitative Imaging Basis
- FOV: 240×200 mm; candidate sensor: 2448×2048.
- Sampling ≈0.098 mm/px → ~5.1 px across 0.5-mm scratch.
- 30 µs exposure at 500 mm/s → low motion-blur target.
- Hardware assumptions require bench verification.

## Slide 4 — Step #2 Synthetic Acquisition
- Deterministic normal/scratch/edge/corner/mixed generation.
- Texture, illumination, blur, noise, markings, pose/translation.
- Exact masks, bounding boxes, JSONL/CSV traceability.
- 70/15/15 train/validation/test separation.

## Slide 5 — Why Hybrid CV + AI?
- Geometry has strong known priors → deterministic CV is explainable and efficient.
- Scratch appearance varies → learned pixel segmentation is appropriate.
- Hybrid design reduces unnecessary model scope and improves traceability.

## Slide 6 — Classical CV Pipeline
- Preprocess/normalize.
- Foreground segmentation and largest contour.
- `minAreaRect` pose estimation.
- Canonical rectification.
- Rectangularity, solidity, corner-count geometry checks.

## Slide 7 — PyTorch Scratch Segmentation
- Compact U-Net: encoder → bottleneck → decoder + skip connections.
- Binary scratch probability map.
- BCE-with-logits + Adam.
- Metrics: precision, recall, Dice/F1, IoU.

## Slide 8 — Label-Integrity Decision
- Step #2 mixed masks combine scratch and chip pixels.
- U-Net therefore trains only on normal + scratch samples.
- Edge/corner/mixed masks are not mislabeled as scratch.
- Future improvement: separate per-defect masks / multi-task labels.

## Slide 9 — Training / Validation / Test Discipline
- Training split updates model weights.
- Validation split selects best epoch/model.
- Held-out test split evaluated only after model selection.
- Fixed random seed supports reproducibility.

## Slide 10 — Hybrid Inference Trace
- Preprocessed image.
- Localization overlay.
- Rectified ROI.
- Scratch probability map.
- Thresholded scratch mask.
- JSON geometry/scratch/decision/latency result.

## Slide 11 — Decision Fusion
- Geometry defect OR scratch detection → REJECT.
- Otherwise → PASS.
- Step #3 demonstrates architecture; Step #4 calibrates/evaluates part-level decisions.

## Slide 12 — Software Engineering
- Modular `src/vision_poc` package.
- YAML configuration, logging, tests, Ruff, CI.
- Module-by-module saved intermediate/final artifacts.
- Reproducible model/data generation.

## Slide 13 — Git/CI Workflow
- Feature #03 isolated from stable/integration branches.
- Ruff + pytest + deterministic CLI smoke test in GitHub Actions.
- PR #1 feature→develop; PR #2 develop→main.

## Slide 14 — Limitations / Domain Gap
- Synthetic BRDF, optics, vibration, contamination and sensor behavior are incomplete.
- Synthetic metrics are feasibility evidence, not factory acceptance.
- Real representative data and formal domain-gap validation remain mandatory.

## Slide 15 — Next: Step #4
- Full held-out end-to-end evaluation.
- Confusion matrix / false accept / false reject.
- Per-defect and defect-size performance.
- Latency distribution and failure gallery.
- PLC/reject-system simulation and integration tests.
