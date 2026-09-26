# End-to-End Machine-Vision System PoC

**Repository:** `mghazel--end-to-end-machine-vision-system-poc`  
**Current release:** Step #4 — End-to-end integration, held-out evaluation and PLC/reject simulation  
**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-25

> **Scope honesty:** Steps #1–#3 are retained cumulatively. Step #4 integrates the hybrid inspector, performs validation-only decision calibration, evaluates the held-out synthetic test split, analyzes latency/failures, and simulates the PLC/reject handoff. Results remain synthetic-domain PoC evidence. No physical camera/PLC or factory-validated acceptance accuracy is claimed.

## 1. Application
Inspect matte-gray 200×120 mm rectangular parts with legitimate text/logos, random ±20° orientation and small translation on a 0.5 m/s conveyor. Target dark scratches/cuts (≥0.5 mm wide, ≥5 mm long), edge damage (≥2 mm), corner damage (≥2 mm), and mixed defects.

## 2. Cumulative architecture
`requirements → hardware engineering → synthetic acquisition → preprocessing → localization/orientation → rectification → classical geometry + PyTorch scratch segmentation → decision fusion → validation calibration → held-out evaluation → PLC/reject simulation → evidence/logging`

See `docs/architecture.md`, `docs/step_03_design.md`, and `docs/step_04_design.md`.

## 3. Five-step roadmap
| Step | Branch | Scope | Status |
|---|---|---|---|
| 1 | `feature/01-requirements-hardware` | Requirements/hardware engineering | Completed |
| 2 | `feature/02-synthetic-dataset` | Synthetic images, masks, annotations, QA | Completed |
| 3 | `feature/03-hybrid-cv-pytorch` | OpenCV localization/geometry + PyTorch U-Net | Completed |
| 4 | `feature/04-integration-evaluation` | Decision calibration, full evaluation, PLC simulation | Implemented |
| 5 | `feature/05-production-readiness` | FAT/SAT, domain gap, risk/maintenance | Planned |

## 4. Step #1 — retained engineering basis
The project retains the 240×200 mm FOV, 2448×2048 candidate camera, 0.5-mm minimum scratch, ≥5 px sampling target, 500 mm/s conveyor, 30 µs exposure and ≤0.2-pixel blur target. These are proposed PoC assumptions and must be verified on the physical station.

## 5. Step #2 — retained synthetic acquisition
The deterministic generator produces normal, scratch, edge-damage, corner-damage and mixed samples with texture, illumination gradient, pose, noise, blur and legitimate markings. It saves exact masks, bounding boxes, JSONL annotations, CSV manifest, deterministic 70/15/15 splits and a QA montage. Default development size is 400 images; a stronger final experiment can increase toward ~4,000 after runtime/storage verification.

## 6. Step #3 — retained hybrid inspection
### Classical CV
`classical_cv.py` performs preprocessing, foreground localization, center/orientation/size estimation, canonical rectification, and deterministic geometry inspection using rectangularity, solidity and corner count.

### PyTorch
`model.py`, `torch_data.py`, `training.py` and `metrics.py` implement a compact U-Net scratch segmenter, train/validation/test execution and pixel-level precision/recall/Dice/IoU. Training uses normal + scratch samples because Step #2 mixed masks do not separate scratch pixels from geometric damage pixels.

### Decision fusion
`hybrid.py` rejects a part when either geometry damage or learned scratch detection is positive and can save each intermediate image plus final JSON result.

## 7. Step #4 — validation-only decision calibration
A key evaluation-control improvement is **test-set isolation**. `evaluation.py` uses the complete validation split to calibrate the scratch positive-area threshold inside the fused part-level decision. Candidate thresholds are swept while geometry decisions remain fixed; balanced accuracy is optimized first, with F1 and then the larger threshold used as tie-breakers.

The selected threshold is then frozen before held-out test evaluation. The test set is not used to tune it.

## 8. Step #4 — held-out part-level evaluation
Ground-truth decision:
- `normal` → PASS
- `scratch`, `edge_damage`, `corner_damage`, `mixed` → REJECT

The test evaluator saves:
- TP/TN/FP/FN;
- accuracy, precision, recall, specificity and F1;
- false-reject rate = normal parts incorrectly rejected;
- false-accept rate = defective parts incorrectly passed;
- per-class correct-decision rates;
- mean, median, p95 and maximum software latency;
- per-part trace CSV;
- confusion matrix;
- latency histogram;
- representative failure gallery.

These metrics quantify the **synthetic PoC**, not production capability.

## 9. Step #4 — PLC/reject-system simulation
`plc.py` demonstrates the production integration contract without pretending to implement a vendor-specific protocol. Every inspected test part receives a unique ID and sequence plus PASS/REJECT, reject output, reason, inspection latency and nominal actuator delay.

For reject distance `d` and conveyor speed `v`:

`reject_delay_ms = 1000 × d / v`

Current PoC values: 750 mm and 500 mm/s → 1500 ms nominal transport delay. A real line should normally use encoder-based tracking if conveyor speed/slip can vary.

## 10. Module-by-module outputs
```text
results/step_04/
  00_step_01_engineering/
    01_metrics/metrics.csv
    02_feasibility/checks.csv
    03_visualizations/*.png
    engineering_results.json
  01_images/{train,validation,test}/
  02_masks/{train,validation,test}/
  03_annotations/annotations.jsonl
  04_qa/{manifest.csv,dataset_montage.png}
  05_classical_cv/classical_cv_summary.json
  06_pytorch/{best_scratch_unet.pt,training_summary.json}
  07_hybrid_trace/
    01_preprocessed.png
    02_localization.png
    03_rectified.png
    04_scratch_probability.png
    05_scratch_mask.png
    06_result.json
  08_integration_evaluation/
    calibration.json
    calibration_curve.png
    evaluation_summary.json
    part_results.csv
    plc_events.json
    confusion_matrix.png
    latency_distribution.png
    failure_gallery.png
  dataset_summary.json
  dataset_report.md
logs/step_04.log
```
Runtime results/models/logs are reproducible and ignored by Git; representative evidence is copied under `docs/sample_results_step_04/` in this release package.

## 11. PyCharm / Windows setup
Python **3.12** is recommended.
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
```
In PyCharm select the repository `.venv` interpreter and repository root as the working directory. The PyTorch dependency is generic; use the appropriate official PyTorch installation for a specific CUDA workstation and verify `torch.cuda.is_available()`.

## 12. Validation gates — run before Git staging
```powershell
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py --skip-training
python scripts/main.py
```
`--skip-training` verifies cumulative engineering, dataset generation and deterministic classical CV quickly. The full command additionally trains/loads the U-Net, calibrates on validation data, evaluates the held-out test split, creates plots/failure evidence and simulates PLC events.

## 13. Git/GitHub — exact copy timing
The detailed tutorial is `docs/github_workflow.md`. The critical order is:

1. Finish/merge Step #3 PRs.
2. Synchronize local `main` and `develop`.
3. Create/switch to `feature/04-integration-evaluation`.
4. Run `git branch` and confirm `*` is on Feature #04.
5. **Only now copy/overwrite this cumulative Step #4 project into the repository.**
6. Install, Ruff, pytest, smoke run, full run.
7. Inspect `git status` and `git diff`.
8. Stage, commit, push.
9. PR #1 feature→develop; green CI; merge.
10. PR #2 develop→main; green CI; merge.
11. Synchronize local branches and stop; create Feature #05 only when Step #5 starts.

Feature commit:
```powershell
git add .
git status
git commit -m "feat(04): integrate evaluation and PLC reject simulation"
git push -u origin feature/04-integration-evaluation
```

## 14. CI/CD
GitHub Actions installs on Python 3.10/3.12, runs Ruff and pytest, generates a small deterministic smoke dataset, executes the non-training cumulative pipeline, and uploads the Step #4 smoke artifact. Unit tests exercise Step #4 metrics and PLC timing without forcing costly model training in CI.

## 15. Engineering interpretation
False accepts and false rejects have different industrial consequences. A false accept sends a defective part downstream/customer; a false reject increases scrap/rework. Threshold selection should therefore eventually use customer-defined costs/acceptance criteria, not only F1. The current validation balanced-accuracy calibration is a transparent PoC policy in the absence of supplied cost weights.

Latency also needs production context. The software p95 is compared with a 200-ms PoC target, but real timing must include exposure/acquisition, image transfer, PLC communication, line tracking and actuator response.

## 16. Limitations and domain gap
Synthetic images do not reproduce the full BRDF, lens MTF/distortion, illumination drift, vibration, contamination, sensor behavior, manufacturing variability or actual defect morphology. A strong synthetic test score can demonstrate software feasibility and integration discipline, but it cannot establish factory accuracy.

## 17. Step #5 preview
Step #5 completes production readiness: real-data/domain-gap plan, FAT/SAT and commissioning protocol, FMEA/risk mitigation, fail-safe behavior, monitoring/traceability, drift and retraining triggers, maintenance/versioning, final BOM/architecture review, final presentation and lessons learned.
