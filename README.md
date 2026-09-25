# End-to-End Machine-Vision System PoC

**Repository:** `mghazel--end-to-end-machine-vision-system-poc`  
**Current release:** Step 1 — Requirements, architecture and quantitative hardware engineering  
**Author:** mghazel  
**Submitted to:** Ascension Automation Solutions Ltd.  
**Version:** 2026-09-25 (the requested “2025” date appears to be a typo; submission context is 2026-09-25)

> **Scope honesty:** This is a functioning **Step 1 engineering design tool**, not yet an operational camera, trained AI model or factory-validated inspection station. Step 2 adds synthetic data; Step 3 adds OpenCV/PyTorch; Step 4 integrates and evaluates; Step 5 addresses commissioning and maintenance. The design parameters are proposed assumptions, not customer-supplied specifications.

## 1. Application

Inspect matte gray 200×120mm rectangular parts with text/logos, random ±20° orientation and ±3mm translation on a 0.5m/s conveyor. Detect dark scratches/cuts (≥0.5mm wide, ≥5mm long), edge chips (≥2mm) and corner chips (≥2mm). Nominal FOV 240×200mm; 450mm part pitch. Targets: ≥5px across minimum scratch, ≤0.2px blur, software processing ≤200ms/part, recall ≥95% and false reject ≤5% (last two require real independent test data).

## 2. Architecture

See [Mermaid architecture diagram](docs/architecture.md). Physical: part/conveyor/trigger/lighting/camera/lens; software: acquisition → preprocessing → localization/rectification → edge/corner CCV + PyTorch scratch segmentation → decision fusion → PLC/result tracking → logging. **Only the configuration/quantitative design/reporting portion is implemented in Step 1.**

## 3. Five cumulative implementation steps

| Step | Feature branch | Scope | Status |
|---|---|---|---|
| 1 | `feature/01-requirements-hardware` | Requirements, optical/motion/throughput engineering, BOM, diagrams, automated checks | Implemented |
| 2 | `feature/02-synthetic-dataset` | Realistic generator, masks, annotations, data QA/splits | Planned |
| 3 | `feature/03-hybrid-cv-pytorch` | OpenCV localization, rectification, edges/corners, PyTorch U-Net | Planned |
| 4 | `feature/04-integration-evaluation` | End-to-end decision, PLC simulator, quantitative evaluation and integration tests | Planned |
| 5 | `feature/05-production-readiness` | FAT/SAT, domain gap, monitoring, risk/maintenance and final presentation | Planned |

## 4. Hardware calculation and selection

### 4.1 Worst-case part envelope

For part width `W=200mm`, height `H=120mm` and angle `θ=20°`:

`W_rot = W cosθ + H sinθ ≈ 229.0mm`; `H_rot = W sinθ + H cosθ ≈ 181.2mm`. Adding ±3mm positional tolerance on each side yields ~235.0×187.2mm; proposed FOV 240×200mm accommodates this but **horizontal margin is narrow**. Confirm actual orientation/translation distributions and consider a larger FOV or guides if needed.

### 4.2 Camera pixel sampling

Minimum object-space sampling = `0.5mm / 5 = 0.1mm/px`. FOV 240×200mm implies ≥2400×2000 pixels. Candidate Basler `a2A2440-98g5mBAS` default 2448×2048, 3.45µm pitch, mono global shutter, 2/3-inch, 5GigE, ~80.8fps at default settings. Result: 240/2448 = 0.09804mm/px; 200/2048 = 0.09766mm/px; a 0.5mm scratch spans ≥5.1px nominally. **Five pixels is a sampling target, not guaranteed detectability**: optics, MTF, SNR, contrast and orientation must be tested.

Manufacturer: https://docs.baslerweb.com/a2a2440-98g5mbas

### 4.3 Lens and working distance

Sensor physical size ≈ `2448×0.00345 = 8.446mm` by `2048×0.00345 = 7.066mm`. Candidate Computar `M1228-MPW3`, 12mm, C-mount, 2/3-inch, 6MP-rated, manufacturer spec: https://cdn.alliedvision.com/fileadmin/content/documents/products/accessories/lenses/Computar/Data_sheet/M1228-MPW3.pdf . Approximate thin-lens working distance from horizontal FOV: `WD ≈ f(1+FOV/sensor_width) ≈ 353mm`; vertical FOV predicts ~352mm. At an assumed nominal 350mm WD, estimated required focal length ≈12.3mm. Verify manufacturer FOV chart, lens distortion, MTF across field, focus range, aperture/DOF and mechanical fit on actual station; no lens performance has been bench verified.

### 4.4 Exposure and lighting

At `v=500mm/s`, a `30µs` exposure creates `500×30×10^-6 = 0.015mm` smear or ~0.154px on the finer image axis. Blur ≤0.2px requires `t ≤ 0.2×0.09766/500 = 39.06µs`. Propose synchronized high-output diffuse LED strobe in an opaque enclosure. **Do not assert adequate SNR until illumination, aperture and actual sensor response are measured.** Surface scratches may benefit from an optional second low-angle dark-field channel in later trials.

### 4.5 Rate and bandwidth

`500mm/s ÷ 450mm pitch ≈1.11 parts/s`, so `900ms` between parts. The 200ms processing budget fits nominal cadence but does **not** prove worst-case latency or correct reject timing. Mono8 raw image `2448×2048≈5.01MB`. At part rate ~5.57MB/s; at rated 80.8fps ~405MB/s, excluding overhead. 5GigE nominal physical link ~625MB/s before protocol/transport losses; benchmark NIC and packet loss at the actual operating mode. Store defects and sampled good images rather than all high-rate frames.

### 4.6 BOM and procurement

See [hardware BOM CSV](docs/hardware_bom.csv). Named camera/lens are **manufacturer-spec candidates**, not purchased or vendor-quoted components. Illumination, trigger, IPC, PLC, NIC, enclosure and rejector are **architecture-level candidates** requiring site-specific engineering and RFQs. No fabricated CAD prices are supplied. The camera is a 5MP model; a 25MP camera is not required by this narrowly specified PoC.

## 5. Module-by-module implementation

- `configuration.py`: validates and loads YAML; rejects missing/invalid sections.
- `engineering.py`: pure, deterministic calculations; checks rotated envelope, pixels/defect, motion blur, camera rate, lens format and timing.
- `reporting.py`: writes JSON, CSV, Markdown and PNG figures to organized result folders.
- `logging_utils.py`: console and file logging.
- `scripts/main.py`: documented driver; orchestrates configuration → engineering → reporting → feasibility status.
- `tests/`: reference, invalid-input, inadequate-camera, overexposure, misalignment and persistence tests.

Every module carries the author, submission and version metadata. Google-style docstrings specify purpose, args, return values and exceptions; the driver documents workflow and exit status.

## 6. PyCharm setup (Windows PowerShell)

1. Extract the ZIP to a local development directory; open its root in PyCharm.
2. Set **Project Interpreter** to Python 3.10+; create a local `.venv`.
3. Open PyCharm Terminal at project root and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/main.py
python -m pytest -q
python -m ruff check src scripts tests
```

For macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate` followed by the same `pip`, `python` commands. In PyCharm create a **Python Run Configuration** with script path `scripts/main.py`, working directory project root, and the project `.venv` interpreter. If `Activate.ps1` is blocked, use `.venv\Scripts\python.exe` directly or change policy only if permitted by your IT environment.

### Expected generated outputs

```text
results/step_01/
  engineering_report.md
  01_calculations/engineering_results.json
  01_calculations/metrics.csv
  02_feasibility/checks.csv
  03_visualizations/scratch_sampling.png
  03_visualizations/feasibility.png
logs/step_01.log
```

Output folders are generated automatically. Runtime results/logs are ignored by Git; reproducible source/config/tests/docs are committed. CI uploads results as downloadable GitHub Actions artifacts.

## 7. Tests and CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs Python 3.10/3.12, editable install, Ruff, pytest, CLI smoke run and artifact upload on feature/develop/main pushes and PRs. Require green CI before merges. **CI is a test/build workflow, not a production deployment of a physical inspection system.** No camera, PLC or PyTorch model is exercised by Step 1 CI.

## 8. GitHub tutorial — initial repository and feature 01

**First-time only:** Create an empty GitHub repository named `mghazel--end-to-end-machine-vision-system-poc` without auto-generated README/.gitignore/license. Use the clean-history workflow below: initialize `main` with a minimal README, create `develop` and the feature branch, and only then extract this complete Step 1 ZIP onto the feature branch. This ensures PR #1 contains the feature diff.

**Recommended clean-history commands (instead of committing full ZIP immediately):**

```bash
# In a NEW empty local directory, before extracting the Step 1 ZIP:
git init -b main
echo '# End-to-End Machine-Vision System PoC' > README.md
git add README.md
git commit -m "chore: initialize project repository"
git remote add origin https://github.com/YOUR_USERNAME/mghazel--end-to-end-machine-vision-system-poc.git
git push -u origin main
git switch -c develop
git push -u origin develop
git switch -c feature/01-requirements-hardware
# NOW extract/copy all ZIP contents into this directory (overwrite README.md).
python -m pip install -e ".[dev]"
python scripts/main.py
python -m pytest -q
python -m ruff check src scripts tests
git add .
git commit -m "feat(01): quantitative vision requirements and hardware design"
git push -u origin feature/01-requirements-hardware
```

Open **Pull Request #1** in GitHub: base `develop`, compare `feature/01-requirements-hardware`. Title: `Feature #01: Requirements, quantitative hardware design and engineering validation`. Description: `Defines inspection assumptions and acceptance goals; implements modular YAML-driven sampling, optics, motion, bandwidth and timing calculations; adds BOM, risk register, charts, tests, CI and cumulative documentation. Validation: pytest, Ruff, CLI. Physical performance remains unverified.` Wait for CI, review, merge (merge commit or squash according to your preference). Pull latest develop locally:

```bash
git switch develop
git pull origin develop
```

Open **Pull Request #2**: base `main`, compare `develop`. Title: `Release Step #01: System requirements and hardware architecture`. Description: `Promotes reviewed Step #01 requirements, calculations, BOM, risk register, reproducible reports and CI from develop to main. CI must pass. No physical inspection hardware or AI model is claimed in this release.` Wait for green CI; merge. Then:

```bash
git switch main
git pull origin main
git switch develop
git pull origin develop
git switch -c feature/02-synthetic-dataset
```

**Branch protection (GitHub Settings → Branches/Rulesets):** Require PR review and passing `validate` checks on `main` and `develop` if your account/repository plan permits. If GitHub reports no differences, verify feature code was committed *after* branching and not already present in `develop`.

## 9. Risk, domain gap, commissioning and next steps

See [risk register](docs/assumptions_risks.md). Synthetic accuracy does not establish real-factory accuracy. Steps 2–5 add data, CV/PyTorch, integration, FAT/SAT planning, commissioning validation and monitoring. Record model/software/config versions, defect-level metrics, failed samples, operational latencies and failure behavior.

## 10. Presentation

See [copy/paste presentation slides](docs/presentation_slides.md), including future-slide placeholders clearly marked as **planned**, not measured results.

## 11. External manufacturer references

- Basler camera documentation: https://docs.baslerweb.com/a2a2440-98g5mbas
- Computar lens datasheet: https://cdn.alliedvision.com/fileadmin/content/documents/products/accessories/lenses/Computar/Data_sheet/M1228-MPW3.pdf

Vendor specs are candidates; price/stock, exact performance, optics and integration need independent confirmation.
