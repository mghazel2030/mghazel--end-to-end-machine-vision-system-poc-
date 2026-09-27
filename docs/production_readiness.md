# Production Readiness, Domain Gap, FAT/SAT and Lifecycle Plan

## Domain-gap strategy
Synthetic data provides exact labels and controlled nuisance variation but cannot reproduce every BRDF, lens MTF/distortion, illumination drift, contamination, vibration, sensor artifact, material lot, logo or plant-background condition. Production qualification therefore requires a representative real-data program.

## Real-data validation
1. Freeze the PoC code/configuration and define data-collection protocol.
2. Acquire independently adjudicated good and defective parts across shifts, lots, speeds and nuisance conditions.
3. Partition real data into adaptation/validation/acceptance sets before tuning.
4. Measure domain shift in intensity/texture/pose/defect-size distributions and failure modes.
5. Adapt preprocessing/model only on designated development data.
6. Freeze model and thresholds; execute independent acceptance testing.
7. Report defect recall, false-accept rate, false-reject rate, latency and defect-size sensitivity with sample counts.

## Fail-safe behavior
Acquisition loss, invalid localization, model/runtime exception, PLC timeout, sequence mismatch or reject-confirmation failure must never silently produce PASS. The production policy should be customer-approved and normally route uncertain states to REJECT, HOLD or controlled line stop with alarm and traceability.

## FAT / SAT / commissioning
The executable generates a detailed acceptance plan under `09_production_readiness/production_readiness.md`. FAT verifies hardware, imaging, golden parts, software release and I/O behavior before shipment. SAT repeats the critical tests at installed line speed and environment using representative production material and independent acceptance data.

## Monitoring, drift and retraining
Trend pass/reject counts, false rejects/escapes, defect-area distributions, pose, illumination/reference statistics, inference latency, camera errors and PLC timeouts. Retraining is triggered by confirmed performance/domain drift—not merely elapsed time—and must pass the same frozen regression/validation/release process.

## Versioning and maintenance
Release artifacts must bind source revision, Python/dependency lock, model checksum, configuration checksum, calibration version and acceptance report. Maintain backups/rollback, preventive optics/lighting cleaning, calibration verification, spare critical hardware and recovery procedures.
