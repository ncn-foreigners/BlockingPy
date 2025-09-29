# Contributing to BlockingPy

Thank you very much for the idea of contributing to BlockingPy!

## Contributing bug fixes, errors, suggestions

Please open an issue [here](https://github.com/T-Strojny/BlockingPy/issues)

# Overview
This repository is a mono-repo with three Python packages:

- packages/blockingpy-core/ – all runtime code, is a common part of both CPU and GPU packages. Not intended to be installed directly (users).

- packages/blockingpy/ – CPU meta‑package. Thin wrapper that depends on core with faiss-cpu.

- packages/blockingpy-gpu/ – GPU meta‑package. Thin wrapper that depends on core. Needs separate installation of faiss-gpu from conda-forge.

Only `blockingpy-core` can be installed as an editable.

Editable install is not available from root of the repo, see below for more information.

## Development instructions
Requirements:

- CPU: Python ≥ 3.10
- Poetry ≥ 1.8
- GPU: Conda environment with Python = 3.10 (required by `faiss-gpu`).

### CPU development (poetry):
```bash
# 1) Clone
git clone https://github.com/ncn-foreigners/BlockingPy.git
cd BlockingPy

# 2) Optional: keep venv in repo
poetry config virtualenvs.in-project true

# 3) Install for CPU development (package + core, editable; with faiss-cpu)
poetry install --with dev,cpu

# 4) Run checks
poetry run pytest -q
poetry run ruff check .
poetry run mypy packages/blockingpy-core
```
### GPU development (conda + poetry)
```bash
# 1) Clone
git clone https://github.com/ncn-foreigners/BlockingPy.git
cd BlockingPy

# 2) Conda env + faiss-gpu
conda create -n blockingpy-gpu python=3.12 -y
conda activate blockingpy-gpu
conda install -c pytorch -c nvidia faiss-gpu -y

# 3) Point Poetry at this same env
poetry config virtualenvs.create false 

# 4) Dev install from repo root
poetry install --with dev,gpu

# 5) Test
poetry run pytest -q
poetry run ruff check .
poetry run mypy 
```

Additional info:
- If FAISS is not present, FAISS‑based tests should skip. If something raises instead of skipping, open an issue.

## Formatting, linting and type checking
We use ruff and mypy:
```bash
poetry run ruff format .
poetry run ruff check .
poetry run mypy .
```

## Tests and coverage
All tests should pass (or skip) no matter if FAISS, MLPACK and faiss-gpu (from conda) are installed or not.
```bash
poetry run pytest -q
poetry run pytest --cov=blockingpy --cov-branch --cov-report=term-missing
```

## Docs
```bash
cd docs
poetry run sphinx-autobuild . ./html
```

## Contributing new feature or change

- Fork the project and create a new branch
- Make changes and commit
- Make sure to add new tests
- Check if both new and old tests pass
- Open a pull request