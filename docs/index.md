# Welcome to BlockingPy's Documentation

[![License](https://img.shields.io/github/license/T-Strojny/BlockingPy)](https://github.com/T-Strojny/BlockingPy/blob/main/LICENSE) 
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Code Coverage](https://img.shields.io/codecov/c/github/T-Strojny/BlockingPy)](https://codecov.io/gh/T-Strojny/BlockingPy)\
[![PyPI version](https://img.shields.io/pypi/v/blockingpy.svg)](https://pypi.org/project/blockingpy/) 
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/T-Strojny/BlockingPy/actions/workflows/run_tests.yml/badge.svg)](https://github.com/T-Strojny/BlockingPy/actions/workflows/run_tests.yml)
[![GitHub last commit](https://img.shields.io/github/last-commit/T-Strojny/BlockingPy)](https://github.com/T-Strojny/BlockingPy/commits/main)

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started/index
user_guide/index
examples/index
api/index
changelog
```

```{include} ../README.md
:start-after: "# BlockingPy"
:end-before: "## Installation"
```
## Key Features

- **Multiple ANN Algorithms**: Supports FAISS, HNSW, Voyager, Annoy, MLPack, and NND
- **Flexible Input**: Works with text data, sparse matrices, or dense feature vectors
- **Customizable Processing**: Configurable text processing and algorithm parameters
- **Performance Focused**: Optimized for both accuracy and computational efficiency
- **Easy Integration**: Simple API that works with pandas DataFrames
- **Quality Assessment**: Built-in evaluation metrics when true matches are known

If you're new to BlockingPy, we recommend following these steps:

1. Start with the {ref}`getting-started` guide to set up BlockingPy
2. Try the {ref}`quickstart` guide to see basic usage examples
3. Look at {ref}`examples` to understand more about BlockingPy
4. Explore the {ref}`user-guide` for detailed usage instructions
5. Obtain more information via {ref}`api`

## Example Datasets

BlockingPy comes with built-in example datasets:

- Census-Cis dataset created by Paula McLeod, Dick Heasman and Ian Forbes, ONS,
    for the ESSnet DI on-the-job training course, Southampton,
    25-28 January 2011

- Deduplication dataset taken from [RecordLinkage](https://cran.r-project.org/package=RecordLinkage) R package developed by Murat Sariyar
    and Andreas Borg. Package is licensed under GPL-3 license. Also known as [RLdata10000](https://www.rdocumentation.org/packages/RecordLinkage/versions/0.4-12.4/topics/RLdata).

## License

BlockingPy is released under [MIT license](https://github.com/ncn-foreigners/BlockingPy/blob/main/LICENSE).

## Issues

Feel free to report any issues, bugs, suggestions with github issues [here](https://github.com/ncn-foreigners/BlockingPy/issues).

## Contributing

Please see [CONTRIBUTING.md](https://github.com/ncn-foreigners/BlockingPy/blob/main/CONTRIBUTING.md) for more information.

## Code of Conduct
You can find it [here](https://github.com/ncn-foreigners/BlockingPy/blob/main/CODE_OF_CONDUCT.md).

## Acknowledgements

This package is based on the R [blocking](https://github.com/ncn-foreigners/blocking/tree/main) package developed by [BERENZ](https://github.com/BERENZ).

## Funding

Work on this package is supported by the National Science Centre, OPUS 20 grant no. 2020/39/B/HS4/00941 (Towards census-like statistics for foreign-born populations -- quality, data integration and estimation)
