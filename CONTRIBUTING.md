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

- GPU: Conda environment with Python = 3.10 (required by `faiss-gpu`).

### CPU development (uv):
```bash
# 1) Clone
git clone https://github.com/ncn-foreigners/BlockingPy.git
cd BlockingPy


# 2) Create venv and activate
uv venv
. .venv/bin/activate # Windows: .venv\Scripts\Activate.ps1


# 3) Editable install of the core + dev tools + FAISS CPU (optional)
uv pip install -e packages/blockingpy-core[dev,faiss]
# uv pip install -e packages/blockingpy-core[dev] # if you don't want faiss


# 4) Test
uv run pytest -q
```
### GPU development (conda + pip)

This mirrors the README: create a conda env first, install the GPU meta‑package with pip inside that env, then install faiss-gpu with conda.
```bash
# 1) Create and activate a conda env with Python 3.10
conda create -n blockingpy-gpu-dev -y python=3.10
conda activate blockingpy-gpu-dev

# 2) Install FAISS GPU into the same conda env
conda install -y -c pytorch -c nvidia faiss-gpu


# 3) Install the editable core + dev tools
python -m pip install -e packages/blockingpy-core[dev]

# 4) Test
pytest -q
```

Additional info:
- If FAISS is not present, FAISS‑based tests should skip. If something raises instead of skipping, open an issue.

## Formatting, linting and type checking
We use ruff and mypy:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy 
```

## Tests and coverage
All tests should pass (or skip) no matter if FAISS, MLPACK and faiss-gpu (from conda) are installed or not.
```bash
uv run pytest -q
uv run pytest --cov=blockingpy --cov-branch --cov-report=term-missing
```

## Docs
```bash
cd docs
uv run sphinx-autobuild . ./html
```

## Contributing new feature or change

- Fork the project and creata new branch
- Make changes and commit
- Make sure to add new tests
- Check if both new and old tests pass
- Open a pull request