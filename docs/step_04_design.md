# Step #4 — End-to-End Integration, Evaluation, and PLC Simulation

**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-25

## 1. Objective
Step #4 converts the Step #3 algorithms into an integrated inspection experiment. It calibrates a decision parameter on validation data, freezes that parameter, runs the hybrid inspector on the held-out test split, quantifies part-level decisions and latency, saves failure cases, and simulates the vision-to-PLC reject handoff.

## 2. Data-integrity rule
The **validation split** may be used for threshold calibration. The **test split is never used to select the threshold**. This avoids optimistic test-set tuning. All resulting performance remains synthetic-domain evidence and must not be presented as factory acceptance performance.

## 3. Decision semantics
Ground truth is part-level:
- `normal` → expected `PASS`;
- `scratch`, `edge_damage`, `corner_damage`, `mixed` → expected `REJECT`.

The hybrid decision remains:
`REJECT = geometry_damage OR scratch_detected`.

The scratch branch first thresholds the U-Net probability map, computes positive-pixel fraction, then compares that fraction with the calibrated validation-set threshold.

## 4. Metrics
Step #4 records TP/TN/FP/FN, accuracy, precision, recall, specificity, F1, false-reject rate, false-accept rate, per-class correct-decision rate, mean/median/p95/max software latency, and the number of PLC events.

Industrial interpretation:
- **False accept (FN):** defective part incorrectly passed; usually the higher product-quality risk.
- **False reject (FP):** normal part incorrectly rejected; drives scrap/rework and throughput cost.
- **p95 latency:** more useful than mean alone for real-time margin assessment.

## 5. PLC/reject simulation
`PLCRejectSimulator` does not emulate a vendor protocol. It demonstrates the integration contract: unique part ID, monotonically increasing sequence, PASS/REJECT, reject output, inspection latency, rejection reason, and nominal transport delay.

For conveyor speed `v` and camera-to-reject distance `d`:

`reject_delay_ms = 1000 * d / v`

At 500 mm/s and 750 mm, nominal delay is 1500 ms. A production implementation should replace this time-only model with encoder-based part tracking where slip/variable speed matter.

## 6. Saved evidence
`08_integration_evaluation/` contains calibration JSON, held-out summary JSON, per-part CSV, PLC event JSON, confusion matrix, latency distribution, and failure gallery. These artifacts support traceability and interview discussion of both successes and failure modes.

## 7. Limitations / Step #5 handoff
No physical I/O, safety PLC, encoder, reject actuator, camera SDK, factory data, FAT/SAT, MSA, or real domain-gap measurement is implemented. Step #5 addresses production readiness, commissioning, domain gap, risk controls, monitoring, and maintenance.
