# High-Level Presentation — cumulative through Step 2

## Slide 1 — Layered system
Physical station → acquisition → preprocessing → CCV geometry → PyTorch
surface segmentation → decision fusion → PLC/reject → logging/traceability.
Step 1 implemented engineering; Step 2 implemented synthetic acquisition/data.

## Slide 2 — Application
200×120 mm matte gray parts, ±20° pose, text/logos; 0.5-mm dark scratches,
2-mm edge/corner damage; one top-down image per part.

## Slide 3 — Step 1 quantitative basis
240×200 mm FOV; 2448×2048 candidate sampling ~0.098 mm/px; >=5 px per
0.5-mm scratch; 500 mm/s conveyor; 30 us exposure; blur ~0.154 px.

## Slide 4 — Step 2 dataset objective
No supplied data → reproducible simulation for software feasibility.
Exact image/mask/class/bounding-box ground truth.

## Slide 5 — Domain randomization
Random surface intensity/texture, illumination gradient, pose, translation,
noise, blur and legitimate markings to reduce synthetic shortcut learning.

## Slide 6 — Dataset summary
| Class | Default generation |
|---|---|
| Normal | balanced cyclic allocation |
| Scratch | balanced cyclic allocation |
| Edge damage | balanced cyclic allocation |
| Corner damage | balanced cyclic allocation |
| Mixed | balanced cyclic allocation |
Default 400 samples; scalable to 4,000 after runtime/storage verification.
70/15/15 train/validation/test.

## Slide 7 — Sample-image panel
Insert `results/step_02/04_qa/dataset_montage.png`.
Red overlay = exact defect ground truth, not detector output.

## Slide 8 — Planned preprocessing / CCV (Step 3)
Quality checks → part contour → orientation → rectification → straight-edge
and 90° corner analysis.

## Slide 9 — Planned DL (Step 3)
PyTorch U-Net: train on synthetic masks; validation for model/threshold
selection; untouched test set for final synthetic benchmark.

## Slide 10 — Planned post-processing/results
Probability → threshold → morphology → connected defects → calibrated
dimensions → decision fusion. Report precision/recall/F1/Dice/IoU.

## Slide 11 — Integration/testing (Step 4)
Part IDs, bounded queues, PLC handshake simulator, latency distributions,
fault injection and end-to-end tests.

## Slide 12 — Deployment/domain gap (Step 5)
Collect representative factory images; quantify synthetic→real degradation;
adapt/retrain; independent real acceptance set; FAT/SAT.

## Slide 13 — Continuous improvement
Hard-negative capture, drift monitoring, controlled model/config versions,
periodic regression testing and illumination/focus health monitoring.

## Slide 14 — Step 2 lessons
Exact synthetic labels accelerate development, but variability must be
deliberate and synthetic accuracy must never be represented as factory accuracy.

## Slide 15 — Next step
Implement hybrid OpenCV + PyTorch inspection while preserving all generated
intermediate results and comparing conventional CV against learned segmentation.
