# Git/GitHub Workflow — Feature #03

## Mental model
- `main`: released/stable project state.
- `develop`: integration branch containing reviewed features intended for the next release.
- `feature/03-hybrid-cv-pytorch`: isolated workspace for Step #3 changes.

## 1. Synchronize after Step #2 PRs are merged
```powershell
git switch main
git pull origin main
git switch develop
git pull origin develop
```
`git switch` changes the checked-out branch. `git pull origin <branch>` downloads the remote branch and integrates its latest commits locally.

## 2. Create Step #3 branch exactly once
```powershell
git switch -c feature/03-hybrid-cv-pytorch
```
`-c` means **create**. If the branch already exists, use `git switch feature/03-hybrid-cv-pytorch` without `-c`.

Confirm before copying files:
```powershell
git branch
git status
```
The `*` must be beside `feature/03-hybrid-cv-pytorch`.

## 3. Copy the cumulative Step #3 ZIP NOW
Extract the Step #3 ZIP to a temporary folder. Copy its contents into the repository root and allow updated files to overwrite their Step #2 versions. Do **not** copy the ZIP before switching to the feature branch. The ZIP is cumulative: Step #1 + corrected Step #2 + new Step #3. Git will calculate the Step #3 delta.

## 4. Install and validate before staging
```powershell
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py --skip-training
python scripts/main.py
```
`pip check` verifies dependency consistency. Ruff performs static quality checks. Pytest runs regression/unit tests. The skip-training command checks the fast deterministic pipeline; the final command performs training and hybrid inference.

## 5. Review what Git sees
```powershell
git status
git diff
```
`git status` lists untracked/modified/staged files. `git diff` shows unstaged line-level changes. Review before staging.

## 6. Stage and commit
```powershell
git add .
git status
git commit -m "feat(03): add hybrid OpenCV and PyTorch inspection pipeline"
```
`git add .` places current changes in the staging area. `git commit` creates the local permanent snapshot from exactly what is staged.

## 7. Publish the feature branch
First push:
```powershell
git push -u origin feature/03-hybrid-cv-pytorch
```
`-u` establishes the upstream relationship. Later pushes from this branch need only `git push`.

## 8. Pull Request #1 — feature → develop
**Base:** `develop`  
**Compare:** `feature/03-hybrid-cv-pytorch`

**Title:** `Feature #03: Hybrid OpenCV geometry inspection and PyTorch scratch segmentation`

**Description:** Implements deterministic preprocessing, part localization, orientation/rectification and geometry inspection; adds compact PyTorch U-Net scratch segmentation, training/validation/test metrics, hybrid PASS/REJECT inference, saved intermediate results, tests, CI updates and cumulative documentation. Synthetic-domain results demonstrate software feasibility only; real factory validation remains required.

Wait for green CI and review, then merge.

## 9. Pull Request #2 — develop → main
First synchronize local develop after PR #1 merge:
```powershell
git switch develop
git pull origin develop
```

**Base:** `main`  
**Compare:** `develop`

**Title:** `Release Step #03: Hybrid CV and PyTorch defect-inspection pipeline`

**Description:** Promotes reviewed Step #03 OpenCV/PyTorch inspection, tests, reproducible outputs, model-training workflow and documentation from develop to main. CI must pass. Results remain synthetic-domain PoC evidence, not factory acceptance metrics.

After PR #2 merges:
```powershell
git switch main
git pull origin main
git switch develop
git pull origin develop
```
Stop here. Create `feature/04-integration-evaluation` only when Step #4 work actually begins.
