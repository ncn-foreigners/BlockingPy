[![License](https://img.shields.io/github/license/ncn-foreigners/BlockingPy)](https://github.com/ncn-foreigners/BlockingPy/blob/main/LICENSE) 
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/ncn-foreigners/BlockingPy/graph/badge.svg?token=BF41O220NY)](https://codecov.io/gh/ncn-foreigners/BlockingPy)\
[![PyPI version](https://img.shields.io/pypi/v/blockingpy.svg)](https://pypi.org/project/blockingpy/) 
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/ncn-foreigners/BlockingPy/actions/workflows/run_tests.yml/badge.svg)](https://github.com/ncn-foreigners/BlockingPy/actions/workflows/run_tests.yml)
[![GitHub last commit](https://img.shields.io/github/last-commit/ncn-foreigners/BlockingPy)](https://github.com/ncn-foreigners/BlockingPy/commits/main)
[![Documentation Status](https://readthedocs.org/projects/blockingpy/badge/?version=latest)](https://blockingpy.readthedocs.io/en/latest/?badge=latest)
![PyPI Downloads](https://img.shields.io/pypi/dm/blockingpy)


# BlockingPy

BlockingPy is a Python package that implements efficient blocking methods for record linkage and data deduplication using Approximate Nearest Neighbor (ANN) algorithms. It is based on [R blocking package](https://github.com/ncn-foreigners/blocking).

## Purpose

When performing record linkage or deduplication on large datasets, comparing all possible record pairs becomes computationally infeasible. Blocking helps reduce the comparison space by identifying candidate record pairs that are likely to match, using efficient approximate nearest neighbor search algorithms.

## Installation

BlockingPy requires Python 3.10 or later. Installation is handled via PIP as follows:
```bash
pip install blockingpy
```
or i.e. with poetry:

```bash
poetry add blockingpy
```
### Note
You may need to run the following beforehand:
```bash
sudo apt-get install -y libmlpack-dev # on Linux
brew install mlpack # on MacOS
```
## Basic Usage
### Record Linkage
```python
from blockingpy import Blocker
import pandas as pd

# Example data for record linkage
x = pd.DataFrame({
    "txt": [
            "johnsmith",
            "smithjohn",
            "smiithhjohn",
            "smithjohnny",
            "montypython",
            "pythonmonty",
            "errmontypython",
            "monty",
        ]})

y = pd.DataFrame({
    "txt": [
            "montypython",
            "smithjohn",
            "other",
        ]})

# Initialize blocker instance
blocker = Blocker()

# Perform blocking with default ANN : FAISS
block_result = blocker.block(x = x['txt'], y = y['txt'])
```
Printing `block_result` contains:

- The method used (`faiss` - refers to Facebook AI Similarity Search)
- Number of blocks created (`3` in this case)
- Number of columns (features) used for blocking (intersecting n-grams generated from both datasets, `17` in this example)
- Reduction ratio, i.e. how large is the reduction of comparison pairs (here `0.8750` which means blocking reduces comparison by over 87.5%).
```python
print(block_result)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 3
# Number of columns created for blocking: 17
# Reduction ratio: 0.8750
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 3  
```
By printing `block_result.result` we can take a look at the results table containing:

- row numbers from the original data,
- block number (integers),
- distance (from the ANN algorithm).

```python
print(block_result.result)
#    x  y  block      dist
# 0  4  0      0  0.000000
# 1  1  1      1  0.000000
# 2  6  2      2  0.607768
```
### Deduplication
We can perform deduplication by putting previously created DataFrame in the `block()` method.
```python
dedup_result = blocker.block(x = x['txt'])
```
```python
print(dedup_result)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 2
# Number of columns created for blocking: 25
# Reduction ratio: 0.571429
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          4 | 2 
```
```python
print(dedup_result.result)
#    x  y  block      dist
# 0  1  0      0  0.125000
# 1  3  1      0  0.105573
# 2  1  2      0  0.105573
# 3  5  4      1  0.083333
# 4  4  6      1  0.105573
# 5  5  7      1  0.278312
```
## Features
- Multiple ANN implementations available:
    - [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) (`lsh`, `hnsw`, `flat`)
    - [Voyager](https://github.com/spotify/voyager) (Spotify)
    - [HNSW](https://github.com/nmslib/hnswlib) (Hierarchical Navigable Small World)
    - [MLPACK](https://github.com/mlpack/mlpack) (both LSH and k-d tree)
    - [NND](https://github.com/lmcinnes/pynndescent) (Nearest Neighbor Descent)
    - [Annoy](https://github.com/spotify/annoy) (Spotify)

- Multiple distance metrics such as:
    - Euclidean
    - Cosine
    - Inner Product
    
    and more...
- Support for both shingle-based and embedding-based text representation
- Comprehensive algorithm parameters customization with `control_ann` and `control_txt`
- Support for already created Document-Term-Matrices (as `np.ndarray` or `csr_matrix`)
- Support for both record linkage and deduplication
- Evaluation metrics when true blocks are known

You can find detailed information about BlockingPy in [documentation](https://blockingpy.readthedocs.io/en/latest/).

## Example Datasets

BlockingPy comes with built-in example datasets:

- Census-Cis dataset created by Paula McLeod, Dick Heasman and Ian Forbes, ONS,
    for the ESSnet DI on-the-job training course, Southampton,
    25-28 January 2011

- Deduplication dataset taken from [RecordLinkage](https://cran.r-project.org/package=RecordLinkage) R package developed by Murat Sariyar
    and Andreas Borg. Package is licensed under GPL-3 license. Also known as [RLdata10000](https://www.rdocumentation.org/packages/RecordLinkage/versions/0.4-12.4/topics/RLdata).


## License
BlockingPy is released under [MIT license](https://github.com/ncn-foreigners/BlockingPy/blob/main/LICENSE).

## Third Party
BlockingPy benefits from many open-source packages such as [Faiss](https://github.com/facebookresearch/faiss) or [Annoy](https://github.com/spotify/annoy). For detailed information see [third party notice](https://github.com/ncn-foreigners/BlockingPy/blob/main/THIRD_PARTY).

## Contributing

Please see [CONTRIBUTING.md](https://github.com/ncn-foreigners/BlockingPy/blob/main/CONTRIBUTING.md) for more information.

## Code of Conduct
You can find it [here](https://github.com/ncn-foreigners/BlockingPy/blob/main/CODE_OF_CONDUCT.md)

## Acknowledgements
This package is based on the R [blocking](https://github.com/ncn-foreigners/blocking/tree/main) package developed by [BERENZ](https://github.com/BERENZ).

## Funding

Work on this package is supported by the National Science Centre, OPUS 20 grant no. 2020/39/B/HS4/00941 (Towards census-like statistics for foreign-born populations -- quality, data integration and estimation)
