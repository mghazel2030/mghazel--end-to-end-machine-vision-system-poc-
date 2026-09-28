# Git/GitHub Workflow — Step #4

## Mental model
- `main`: released/accepted project state.
- `develop`: integrated development baseline.
- `feature/04-integration-evaluation`: isolated workspace for Step #4.
- A commit is a local snapshot. A push publishes commits. A pull request asks GitHub to review/merge one branch into another.

## 1. Synchronize before starting
After Step #3 PR #1 and PR #2 are merged:
```powershell
git switch main
git pull origin main
git switch develop
git pull origin develop
```
`git switch` changes the checked-out branch. `git pull origin <branch>` updates that local branch from GitHub.

## 2. Create the Step #4 feature branch
```powershell
git switch -c feature/04-integration-evaluation
```
Use `-c` only when the branch does **not** already exist. If it exists:
```powershell
git switch feature/04-integration-evaluation
```
Verify before copying any files:
```powershell
git branch
git status
```
The `*` must be beside `feature/04-integration-evaluation`.

## 3. Copy the cumulative Step #4 project — exactly here
Only **after** Feature #04 is active, extract the supplied ZIP to a temporary folder and copy its contents over the repository root, allowing updated files to overwrite prior versions. Do not delete `.git`. Do not copy a `.venv` from another machine.

Why timing matters: Git compares the Step #4 working tree with the Step #3 commit inherited by the feature branch. If you copy while on `develop` or `main`, the uncommitted Step #4 changes are attached to the wrong working branch and the intended feature PR history becomes confusing.

## 4. Install and validate before staging
```powershell
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py --skip-training
python scripts/main.py
```
`pip check` validates installed dependency consistency. Ruff is static source quality. Pytest is regression/unit testing. The skip-training run is a fast deterministic smoke path. The final run exercises training, calibration, held-out evaluation, and PLC simulation.

## 5. Inspect changes before committing
```powershell
git status
git diff
```
`git status` shows changed/untracked files. `git diff` shows unstaged line-level changes. Review both before staging.

## 6. Stage and commit
```powershell
git add .
git status
git commit -m "feat(04): integrate evaluation and PLC reject simulation"
```
`git add .` stages the reviewed working-tree changes. The second `git status` shows exactly what the commit will contain. `git commit` records that staged snapshot locally.

## 7. Publish Feature #04
First push:
```powershell
git push -u origin feature/04-integration-evaluation
```
`-u` establishes the upstream relationship. Later pushes from this branch can normally use only `git push`.

## 8. Pull Request #1 — feature → develop
Base: `develop`  
Compare: `feature/04-integration-evaluation`

**Title:** `Feature #04: End-to-end evaluation, decision calibration and PLC reject simulation`

**Description:**
`Integrates the hybrid inspector across held-out synthetic data; calibrates the scratch decision threshold on validation data only; adds part-level TP/TN/FP/FN, false-accept/false-reject and latency metrics, per-class results, failure-case visualization, PLC/reject-event simulation, tests, CI updates and cumulative documentation. Results remain synthetic-domain PoC evidence, not factory acceptance metrics.`

Wait for green CI, review the Files Changed tab, then merge.

## 9. Pull Request #2 — develop → main
After PR #1:
```powershell
git switch develop
git pull origin develop
```
Base: `main`  
Compare: `develop`

**Title:** `Release Step #04: Integrated inspection evaluation and reject-system simulation`

**Description:**
`Promotes reviewed Step #04 integration/evaluation functionality from develop to main, including validation-only calibration, held-out test metrics, latency/failure analysis, PLC event simulation, automated tests and documentation. Physical PLC/camera integration and factory acceptance remain Step #05/commissioning activities.`

## 10. Synchronize after release — then stop
```powershell
git switch main
git pull origin main
git switch develop
git pull origin develop
```
Do **not** create Feature #05 yet. Create it only when Step #5 begins.

## Command reference
| Command | Meaning | Typical use |
|---|---|---|
| `git status` | Show working-tree/staging state | Before and after staging |
| `git branch` | List branches; `*` marks active branch | Before copying a new step |
| `git switch X` | Switch to existing branch X | Navigation |
| `git switch -c X` | Create and switch to new branch X | Once per new feature |
| `git pull origin X` | Fetch + integrate remote X locally | After merges/before branching |
| `git diff` | Show unstaged content changes | Code review before staging |
| `git add .` | Stage changes | Before commit |
| `git commit -m ...` | Save staged snapshot locally | After validation |
| `git push -u origin X` | First push + set upstream | First publication of branch |
| `git push` | Publish later local commits | Subsequent updates |

## Final Refinement Feature — 2026-09-27

Use `feature/06-additional-refinements` for the presentation-evidence and expanded-evaluation changes.

```powershell
git switch develop
git pull origin develop
git switch -c feature/06-additional-refinements
```

Create the branch **before** copying the cumulative refinement package over the repository root. This ensures Git records the final committed baseline as the parent and shows only the refinement delta in the pull request.

After installation and validation:

```powershell
git status
git diff
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py --skip-training
python scripts/main.py
git add .
git status
git commit -m "feat(06): add presentation evidence and expanded evaluation"
git push -u origin feature/06-additional-refinements
```

Open PR #1 from `feature/06-additional-refinements` to `develop`. After CI passes and the PR is merged, synchronize `develop`, then open PR #2 from `develop` to `main` for the final release.
