# End-to-End Machine-Vision System PoC

**Repository:** `mghazel--end-to-end-machine-vision-system-poc`  
**Current release:** Step 2 — Reproducible synthetic dataset, exact labels and QA  
**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-25

> **Scope honesty:** Step 1 engineering is retained cumulatively. Step 2 adds
> executable synthetic data generation. No trained PyTorch model, physical
> camera test, PLC integration or factory-validated accuracy is claimed yet.

## 1. Application
Inspect matte gray 200×120 mm rectangular parts with legitimate text/logos,
random ±20° orientation and small translation on a 0.5 m/s conveyor. Target
dark scratches/cuts (>=0.5 mm wide, >=5 mm long), edge damage (>=2 mm) and
corner damage (>=2 mm). Step 1 established the quantitative imaging baseline.

## 2. Cumulative architecture
See `docs/architecture.md`. Step 2 inserts a deterministic synthetic acquisition
layer between engineering requirements and future CV/DL development:
requirements → hardware engineering → synthetic scene/defect generation →
images/masks/annotations → splits/QA → future OpenCV/PyTorch.

## 3. Five-step roadmap
| Step | Branch | Scope | Status |
|---|---|---|---|
| 1 | `feature/01-requirements-hardware` | Requirements/hardware engineering | Completed |
| 2 | `feature/02-synthetic-dataset` | Synthetic images, masks, annotations, QA/splits | Implemented |
| 3 | `feature/03-hybrid-cv-pytorch` | OpenCV + PyTorch U-Net | Planned |
| 4 | `feature/04-integration-evaluation` | End-to-end decision/PLC/evaluation | Planned |
| 5 | `feature/05-production-readiness` | FAT/SAT/domain gap/maintenance | Planned |

## 4. Step 1 retained engineering basis
The cumulative code re-runs FOV, sampling and motion feasibility before
generating data. Current design assumptions remain 240×200 mm FOV,
2448×2048 camera, 0.5-mm scratch, >=5 pixels/scratch, 500 mm/s conveyor,
30-us exposure and <=0.2-pixel motion-blur target. These are PoC assumptions,
not employer-supplied acceptance specifications.

## 5. Step 2 implementation
### 5.1 Synthetic scene
`synthetic.py` creates a gray background and matte rectangular part, then
randomizes base reflectance, low-frequency texture, illumination gradient,
pose/translation, Gaussian noise, blur and legitimate alphanumeric markings.

### 5.2 Defects and ground truth
- Normal: no defect mask.
- Scratch: dark, oriented narrow line with exact mask.
- Edge damage: removed material at a nominal edge.
- Corner damage: removed material at a nominal corner.
- Mixed: scratch plus geometric damage.
Masks undergo the same pose transform as the image. Bounding boxes are derived
from final nonzero mask pixels.

### 5.3 Dataset split and reproducibility
The default run creates **400** images for fast development. Class assignment is
balanced cyclically; split is deterministic 70/15/15. Increase to ~4,000 before
Step 3 final training after confirming local runtime/storage. Every sample uses
`base_seed + index`, enabling exact regeneration. Never tune on the test set.

### 5.4 Output organization
```text
results/step_02/
  00_step_01_engineering/engineering_results.json
  01_images/{train,validation,test}/part_*.png
  02_masks/{train,validation,test}/part_*.png
  03_annotations/annotations.jsonl
  04_qa/manifest.csv
  04_qa/dataset_montage.png
  dataset_summary.json
  dataset_report.md
logs/step_02.log
```
Intermediate/final artifacts are named and segregated by module as required.

## 6. Module structure
```text
src/vision_poc/
  configuration.py   YAML loading/validation
  engineering.py     cumulative Step 1 calculations
  logging_utils.py   console/file logging
  synthetic.py       image/defect simulator
  dataset.py         split, persistence, annotations, QA
  reporting.py       cumulative reports
scripts/main.py       documented orchestration driver
tests/                engineering/generator/dataset tests
docs/                 architecture, design, slides, Git workflow
```
Functions include purpose, parameters, return/exception documentation; files
carry author, submission and version metadata.

## 7. PyCharm setup and execution
Recommended interpreter: **Python 3.12**.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py
```
In PyCharm use project `.venv`, script `scripts/main.py`, working directory
project root. `opencv-python-headless` is intentional: generation/reporting
does not need GUI windows and remains CI-friendly.

## 8. Testing
Tests verify cumulative Step 1 feasibility, deterministic sample regeneration,
normal/defect mask semantics, image persistence and a 20-image end-to-end
dataset smoke pipeline. CI runs Ruff + pytest + a 20-image CLI smoke run on
Python 3.10/3.12 and uploads Step 2 artifacts.

## 9. CI/CD
`.github/workflows/ci.yml` runs on feature/develop/main pushes and PRs.
This is build/test automation, **not deployment of a physical inspection
station**. Green CI is required before both merges.

## 10. GitHub Feature #02
Starting from synchronized `main` and `develop`:
```powershell
git switch develop
git pull origin develop
git switch -c feature/02-synthetic-dataset
```
Copy this cumulative ZIP's contents over the repository root, then:
```powershell
python -m pip install -e ".[dev]"
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py
git add .
git commit -m "feat(02): add reproducible synthetic defect dataset pipeline"
git push -u origin feature/02-synthetic-dataset
```
PR #1: feature → develop. PR #2: develop → main. Exact titles/descriptions and
post-merge commands are in `docs/github_workflow.md`.

## 11. Dataset QA and interpretation
Open `results/step_02/04_qa/dataset_montage.png`. Red pixels visualize **known
synthetic ground truth**, not an AI prediction. Inspect class balance,
marking/texture variation, defect visibility and mask alignment. The JSONL and
CSV files provide traceability for every generated sample.

## 12. Domain gap and risk
Synthetic images cannot reproduce all surface BRDF, optical MTF/distortion,
illumination nonuniformity, vibration, contamination or sensor behavior.
Therefore synthetic test metrics in later steps establish software feasibility,
not factory acceptance. Step 5 must collect representative real parts and use
an independent real validation/acceptance set.

## 13. Presentation
`docs/presentation_slides.md` contains 15 cumulative copy-ready slides:
system layers, requirements, Step 1 calculations, Step 2 dataset design and
sample montage, planned CV/DL pipeline, integration, domain gap, maintenance,
summary, lessons and future work.

## 14. Lessons learned
Synthetic generation is strongest when nuisance variation is explicit and
labels are exact. Reproducibility and traceability matter as much as image
count. The test split is a protected evaluation resource. Real-data validation
remains mandatory.

## 15. Step 3 preview
Implement part localization/orientation/rectification, deterministic edge and
corner inspection, PyTorch U-Net scratch segmentation, saved intermediate
results, CCV-vs-DL comparison and quantitative validation.
