# Feature #02 Git/GitHub workflow

Starting condition: Feature #01 is already merged through `develop` into `main`.

```powershell
git switch main
git pull origin main
git switch develop
git pull origin develop
git switch -c feature/02-synthetic-dataset
```

Extract/copy the **contents** of the Step 2 cumulative ZIP into the existing
repository root (do not create a nested repository directory), then:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check src scripts tests
python -m pytest -q
python scripts/main.py
git status
git add .
git commit -m "feat(02): add reproducible synthetic defect dataset pipeline"
git push -u origin feature/02-synthetic-dataset
```

## Pull Request #1
Base: `develop`  
Compare: `feature/02-synthetic-dataset`  
Title: **Feature #02: Synthetic industrial-part dataset, labels and QA**

Description:
- Adds deterministic synthetic conveyor-part generator.
- Adds normal, scratch, edge-damage, corner-damage and mixed samples.
- Saves exact masks, bounding boxes, JSONL annotations and CSV manifest.
- Adds 70/15/15 splits, QA montage, logging, tests and cumulative docs.
- Preserves Step 1 engineering validation.
- Synthetic data is not claimed as real-factory validation.

Wait for `validate` CI to pass, review Files Changed, then merge.

```powershell
git switch develop
git pull origin develop
```

## Pull Request #2
Base: `main`  
Compare: `develop`  
Title: **Release Step #02: Reproducible synthetic dataset and ground truth**

Description:
Promotes reviewed Step #02 generator, annotations, QA, tests and cumulative
documentation to main. Step 3 will implement OpenCV localization/geometry and
PyTorch scratch segmentation.

After green CI and merge:
```powershell
git switch main
git pull origin main
git switch develop
git pull origin develop
git switch -c feature/03-hybrid-cv-pytorch
```
