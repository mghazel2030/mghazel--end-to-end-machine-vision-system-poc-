# High-level presentation — Copy/paste into PowerPoint

**Author:** mghazel · **Submitted to:** Ascension Automation Solutions Ltd. · **Version:** 2026-09-25

## Slide 1 — Project and purpose
- End-to-end hybrid machine-vision inspection for oriented rectangular conveyor parts.
- Defects: dark scratches/cuts, damaged straight edges, damaged right-angle corners.
- Step 1: requirements, architecture, quantitative hardware design, risks and automated checks.

## Slide 2 — System layers (one-slide architecture)
- Physical: conveyor → trigger → strobe → mono global-shutter camera + lens.
- Acquisition: camera SDK abstraction + part ID + timestamp (Step 2 onward).
- Image: preprocessing → localization → rotation correction → rectification (Step 3).
- Hybrid inspection: deterministic edges/corners + PyTorch scratch segmentation (Step 3).
- Automation: decision fusion → PLC handshake → reject confirmation (Step 4).
- Operations: QA metrics, logs, FAT/SAT, domain-gap and drift monitoring (Steps 4–5).

## Slide 3 — Application and assumptions
- Part 200×120 mm, ±20° rotation, ±3 mm position, matte gray, optional text/logos.
- FOV 240×200 mm; conveyor 0.5 m/s; pitch 450 mm.
- Minimum scratch width 0.5 mm; edge/corner damage 2 mm.
- Assumptions proposed for PoC, not supplied by customer.

## Slide 4 — Acceptance requirements
- Target ≥5 pixels across 0.5mm scratch and ≤0.2 pixel motion blur.
- Target recall ≥95%; false reject ≤5%; 200ms software processing budget.
- These are unvalidated goals; real factory data required for production acceptance.

## Slide 5 — Camera and sampling calculations
- Candidate Basler a2A2440-98g5mBAS: 2448×2048, 3.45µm, 2/3-inch, mono/global/5GigE.
- Object sampling: 240/2448 = 0.0980mm/px; 200/2048 = 0.0977mm/px.
- 0.5mm scratch: 5.10 pixels horizontally and 5.12 vertically.
- Rotated envelope at 20° ≈229×181mm; verify ±3mm positional tolerance.

## Slide 6 — Lens, exposure and throughput
- Candidate Computar M1228-MPW3 12mm 2/3-inch C-mount; thin-lens WD ≈353mm horizontal.
- At 0.5m/s and 30µs, blur 0.015mm ≈0.154px; 0.2px limit ≈39µs.
- 450mm pitch / 500mm/s = 0.9s/part ≈1.11 parts/s.
- Full-frame Mono8 ≈5.01MB; nominal 80.8fps ≈405MB/s; actual part rate ≈5.57MB/s.
- Bench-verify light energy, MTF, FOV, distortion and trigger timing.

## Slide 7 — Planned data and model (Step 2–3; NOT YET RESULTS)
| Split | Planned share | Planned count for 4000 images |
|---|---:|---:|
| Train | 70% | 2800 |
| Validation | 15% | 600 |
| Test | 15% | 600 |
- Sample image panels: normal, scratch, edge chip, corner chip, mixed (to be generated Step 2).
- PyTorch U-Net scratch mask + CCV geometry, then hybrid decision fusion.

## Slide 8 — Planned preprocessing, CCV and DL
- Preprocessing: illumination correction, ROI, denoise, calibration and rectification.
- CCV: contour/line fitting, expected rectangle and edge/corner deviation.
- DL: labeled synthetic masks, augmentation, U-Net training, checkpointing, blind test.
- Postprocessing: confidence/geometry filtering, connected components, physical measurements.

## Slide 9 — Planned results and evaluation (Step 3–4)
- Save raw/rectified/edge/corner/AI probability/AI mask/final overlay per part.
- Report pixel Dice/IoU, part-level confusion matrix, precision/recall, false rejects and latency.
- Include real failure examples and threshold trade-offs; do not present synthetic metrics as factory metrics.

## Slide 10 — Integration and end-to-end testing (Step 4)
- Simulate trigger → part ID → capture → hybrid inference → PLC result/ack.
- Inject timeouts, missing frames, full queues, late results and wrong part IDs.
- Test determinism, latency and traceability.

## Slide 11 — Deployment, domain gap and maintenance (Step 5)
- FAT/SAT, real-data blind validation, camera/lighting setup, calibration, PLC HIL.
- Monitor light drift, focus, false rejects, model drift, software/model versions.
- Real-domain commissioning is mandatory before production acceptance.

## Slide 12 — Summary, lessons learned, future work
- Step 1 completed: quantitative design, modular code, automated feasibility checks, CI.
- Engineering lesson: image formation and part tracking precede algorithm optimization.
- Next: synthetic dataset and image acquisition simulation; then hybrid CV/PyTorch.
- Future: real industrial dataset, optical bench, PLC hardware, long-duration trials.
