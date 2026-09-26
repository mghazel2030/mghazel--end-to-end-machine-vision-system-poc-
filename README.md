# End-to-End Machine-Vision System PoC

**Repository:** `mghazel--end-to-end-machine-vision-system-poc`  
**Current release:** Step #3 — Hybrid OpenCV + PyTorch inspection pipeline  
**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-25

> **Scope honesty:** Steps #1–#2 are retained cumulatively. Step #3 adds executable classical CV and PyTorch segmentation. All current image/model performance is synthetic-domain PoC evidence. No physical camera, PLC, or factory-validated acceptance accuracy is claimed.

## 1. Application
Inspect matte gray 200×120 mm rectangular parts with legitimate text/logos, random ±20° orientation and small translation on a 0.5 m/s conveyor. Target dark scratches/cuts (≥0.5 mm wide, ≥5 mm long), edge damage (≥2 mm), and corner damage (≥2 mm).

## 2. Cumulative architecture
`requirements → hardware engineering → synthetic acquisition → preprocessing → part localization/orientation → rectification → classical geometry inspection + PyTorch scratch segmentation → decision fusion → trace/results`

See `docs/architecture.md` and `docs/step_03_design.md`.

## 3. Five-step roadmap
| Step | Branch | Scope | Status |
|---|---|---|---|
| 1 | `feature/01-requirements-hardware` | Requirements/hardware engineering | Completed |
| 2 | `feature/02-synthetic-dataset` | Synthetic images, masks, annotations, QA | Completed |
| 3 | `feature/03-hybrid-cv-pytorch` | OpenCV localization/geometry + PyTorch U-Net | Implemented |
| 4 | `feature/04-integration-evaluation` | Decision/PLC simulation/full evaluation | Planned |
| 5 | `feature/05-production-readiness` | FAT/SAT/domain gap/maintenance | Planned |

## 4. Step #1 retained engineering basis
The project retains the 240×200 mm FOV, 2448×2048 candidate camera, 0.5-mm minimum scratch, ≥5 px sampling target, 500 mm/s conveyor, 30 µs exposure and ≤0.2-pixel blur target. These are PoC assumptions, not customer-supplied acceptance specifications.

## 5. Step #2 retained synthetic acquisition
The deterministic generator creates normal, scratch, edge-damage, corner-damage, and mixed samples with texture, illumination gradient, pose, noise, blur, markings, masks, bounding boxes, JSONL annotations, CSV manifest, split assignment, and QA montage. Default development size remains 400 images. Increase toward ~4,000 for a stronger final training experiment after validating local runtime/storage.

## 6. Step #3 classical CV
`src/vision_poc/classical_cv.py` implements:
- grayscale denoising/normalization;
- bright-part foreground segmentation;
- largest-contour localization;
- center/orientation/size estimation with `cv2.minAreaRect`;
- canonical rotation/cropping;
- geometry inspection using rectangularity, solidity, and corner count;
- localization visualization.

Classical CV is intentionally used where the part geometry provides strong deterministic priors.

## 7. Step #3 PyTorch segmentation
`src/vision_poc/model.py` implements a compact U-Net with two encoder levels, bottleneck, skip connections, and a binary segmentation head. `training.py` performs deterministic train/validation/test execution using BCE-with-logits loss and Adam.

The AI training dataset deliberately uses only **normal + scratch** samples. Step #2 stores a single combined mask for mixed defects, so mixed masks cannot distinguish scratch from chip pixels. Excluding mixed/geometry classes prevents incorrect scratch labels. A future generator should export per-defect masks for multi-task learning.

Metrics: pixel precision, recall, Dice/F1, IoU, train loss, validation loss, and held-out test loss/metrics. The test set is not used for model selection.

## 8. Hybrid inference
`src/vision_poc/hybrid.py` executes preprocessing → localization → geometry → rectification → U-Net probability → scratch mask → decision fusion. A part is rejected if deterministic geometry damage or learned scratch detection is positive. Step #4 will perform full dataset-level decision calibration and evaluation rather than claiming production thresholds in Step #3.

## 9. Module-by-module saved outputs
```text
results/step_03/
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
  dataset_summary.json
  dataset_report.md
logs/step_03.log
```
Runtime results/models/logs are reproducible and ignored by Git. CI uploads its smoke-test result artifact.

## 10. PyCharm / Windows setup
Python **3.12** is recommended.
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
```
In PyCharm, select the repository `.venv` interpreter and project root as working directory. PyTorch in `pyproject.toml` is intentionally generic; if a specific NVIDIA CUDA build is desired, use the installation command appropriate to the workstation/CUDA stack and verify `torch.cuda.is_available()`.

## 11. Validation gates
Run these **before Git staging/commit**:
```powershell
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py --skip-training
python scripts/main.py
```
The first two are mandatory source/regression gates. `--skip-training` quickly verifies cumulative engineering, synthetic generation, and classical CV. The final command performs training, held-out evaluation, model save, and hybrid inference.

## 12. Git/GitHub — exact copy timing
Read `docs/github_workflow.md`. The key rule is: **first synchronize `main`/`develop`, then create/switch to `feature/03-hybrid-cv-pytorch`, verify the `*` is on that feature branch, and only then copy/overwrite the cumulative Step #3 ZIP into the repository root.** Copying earlier would put Step #3 changes on the wrong branch.

Feature commit:
```powershell
git add .
git commit -m "feat(03): add hybrid OpenCV and PyTorch inspection pipeline"
git push -u origin feature/03-hybrid-cv-pytorch
```
PR #1 is feature→develop; after merge, PR #2 is develop→main. Do not create Feature #04 until Step #4 actually begins.

## 13. CI/CD
GitHub Actions installs the package on Python 3.10/3.12, runs Ruff and pytest, generates a 20-image synthetic smoke dataset, runs the Step #3 deterministic pipeline with `--skip-training`, and uploads results. PyTorch architecture/metric behavior is exercised by pytest without making CI perform a costly training job.

## 14. Interpretation and limitations
Synthetic validation can demonstrate architecture, software correctness, traceability, and relative model behavior, but cannot establish factory accuracy. Real deployment requires representative real images, optical/calibration verification, domain-gap measurement, validation-set threshold calibration, independent acceptance data, FAT/SAT, and monitoring.

## 15. Step #4 preview
Step #4 will run the hybrid inspector across the held-out dataset, add part-level TP/TN/FP/FN, per-defect confusion metrics, false-accept/false-reject rates, latency distributions, failure-case galleries, decision calibration, and a PLC/reject-system simulator.
