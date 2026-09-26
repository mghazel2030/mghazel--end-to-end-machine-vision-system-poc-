# Copy/Paste Presentation Slides — Cumulative Through Step #4

## Slide 1 — End-to-End Industrial Machine-Vision PoC
- Conveyor inspection of 200×120 mm matte-gray parts
- Defects: scratches/cuts, edge damage, corner damage, mixed defects
- Hybrid deterministic CV + PyTorch architecture
- Cumulative engineering → simulation → inspection → integration/evaluation

## Slide 2 — System Layers
- Physical: conveyor, trigger/encoder, illumination, lens, camera, IPC/GPU, PLC/reject
- Acquisition/simulation: traceable images, labels, nuisance variation
- CV: preprocessing, localization, pose estimation, rectification, geometry inspection
- AI: compact U-Net scratch segmentation
- Integration: decision fusion, part tracking, PLC event, logging/results

## Slide 3 — Quantitative Imaging Basis
- FOV: 240×200 mm; candidate 2448×2048 mono global-shutter camera
- Sampling ≈0.098 mm/px; 0.5-mm scratch ≈5.1 px
- Conveyor 500 mm/s; exposure 30 µs; nominal motion blur ≈0.15 px
- Hardware assumptions require physical verification before production

## Slide 4 — Synthetic Dataset
- Normal, scratch, edge, corner and mixed classes
- Texture, illumination, rotation/translation, blur/noise, legitimate markings
- Exact masks/bounding boxes and deterministic seed-based regeneration
- 70/15/15 train/validation/test split; default development set 400 images

## Slide 5 — Classical CV Pipeline
- Denoise/normalize
- Bright-part segmentation and largest contour
- `minAreaRect` center/orientation/size
- Canonical rectification
- Rectangularity, solidity and corner-count geometry checks

## Slide 6 — PyTorch Scratch Segmentation
- Compact U-Net: encoder/bottleneck/decoder + skip connections
- BCE-with-logits loss + Adam
- Pixel precision, recall, Dice/F1 and IoU
- Normal + scratch only because Step #2 mixed mask does not isolate defect types

## Slide 7 — Step #4 Integration Architecture
- Hybrid result = deterministic geometry OR learned scratch decision
- Unique part record → decision → PLC/reject event
- Module-by-module traceability from image through actuator command
- Explicit exception/logging path for failed processing

## Slide 8 — Threshold Calibration Without Test Leakage
- Scratch-area threshold calibrated on validation split only
- Sweep candidate thresholds and maximize validation balanced accuracy
- Freeze selected threshold before test evaluation
- Held-out test split remains independent evaluation evidence

## Slide 9 — Part-Level Evaluation
- TP: defective/rejected; TN: normal/passed
- FP: normal/rejected → false reject / production cost
- FN: defective/passed → false accept / quality risk
- Report accuracy, precision, recall, specificity, F1, FAR and FRR

## Slide 10 — Per-Defect Analysis
- Normal acceptance rate
- Scratch detection rate
- Edge-damage detection rate
- Corner-damage detection rate
- Mixed-defect detection rate
- Failure cases retained for root-cause analysis rather than hidden

## Slide 11 — Real-Time Performance
- Measure every held-out inspection latency
- Report mean, median, p95 and maximum
- Compare p95 against 200-ms PoC software target
- Production timing must include acquisition, communications, PLC and actuator margins

## Slide 12 — PLC / Reject Simulation
- Ordered sequence and unique part ID
- PASS/REJECT output plus reason
- Camera-to-reject transport delay: `1000*d/v`
- Example: 750 mm / 500 mm/s = 1500 ms
- Production recommendation: encoder tracking for variable speed/slip

## Slide 13 — Evidence Saved Automatically
- Calibration sweep/selected threshold
- Per-part CSV and PLC-event JSON
- Confusion matrix and latency histogram
- Failure gallery
- Model, training summary, intermediate hybrid trace, logs

## Slide 14 — What Step #4 Demonstrates / Does Not Demonstrate
- Demonstrates integrated software architecture, traceability, quantitative synthetic evaluation
- Demonstrates validation/test separation and deterministic reject-system contract
- Does not establish factory accuracy or physical PLC/camera performance
- Real images, FAT/SAT, MSA, domain-gap validation and commissioning remain mandatory

## Slide 15 — Step #5: Production Readiness
- Domain-gap acquisition and adaptation plan
- FAT/SAT and commissioning protocol
- FMEA/risk mitigation and fail-safe behavior
- Monitoring, model/config versioning, drift and maintenance
- Final architecture, lessons learned and future improvements
