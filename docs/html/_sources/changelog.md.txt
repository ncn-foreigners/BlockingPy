# Changelog

## v0.2.8
- improved reproducibility
- improved logging

## v0.2.7
- new release with zenodo activation

## v0.2.6
- fixed minor issue when running tests with `blockingpy-core`
- improved error handling when `faiss` is not installed

## v0.2.5
- fixed issue in the publish workflow

## v0.2.4
- improved logging and warnings across the package
- improved coverage of tests
- updated API section
- updated README files
- added linting, formatting, and type checking workflow
- improved tests workflow
- datasets are now fetched via Pooch instead of being included in the package
- minor changes

## v0.2.3
- minor fixes in workflows

## v0.2.2
- minor fixes in workflows

## v0.2.1
- minor fixes in workflows

## v0.2.0
- ***GPU SUPPORT*** - through `faiss_gpu`, available indexes (`flat`, `ivf`, `ivfpq`, `cagra`)
- Custom `DataHandler` class to wrap data replacing pandas dataframe
- Switched from `networkx` to `igraph` for graph handling
- Major memory & speed improvements for the CPU `BlockingPy`
- Minor fixes & improvements

## v0.1.15
- added embedding-based encoding support via `model2vec` library
- fix `lowercase=False` bug
- added `add_block_column` method to `BlockingResult` class
- optimized evaluation
- added `controls_txt` validation

## v0.1.14
- changed defaul `lsh_nbits` from 8 to 2
- improved confusion matrix
- minor changes

## v0.1.13
- set default `random_seed` to 2025
- added `IndexLSH` and `IndexHNSWFlat` to `faiss`

## v0.1.12
- impoved reproducibility of the results
- added `random_seed` parameter to `Blocker` class
- new&improved examples in docs
- minor fixes

## v.0.1.11
- recordlinkage package integration example in docs
- minor fixes

## v0.1.10
- evaluation only for records that exist in true blocks.
- default distance for `faiss` changed to `cosine`
- minor changes

## v0.1.9
- optimized evaluation part to allow batch processing

## v0.1.8 
- added author Maciej Beręsewicz
- added info about funding
- added data inside the package
- added new deduplication example in docs
- minor changes

## v0.1.7
- added CODE_OF_CONDUCT.md
- documentation update
- fixed issus with inner ANN algorithms when performing deduplication

## v0.1.6
- revamped block size distribution calculation.

## v0.1.5
- added separate `eval` method strictly for evaluation.
- allowed for `from blockingpy import Blocker` instead of `from blockingpy.blocker import Blocker`

## v0.1.4
- fixed reduction ratio calculation
- new evaluation system for record linkage
- new "evaluation" section in documentation
- changed records filtering system for deduplication
- updated confusion matrix
- minor changes

## v0.1.3

- Initial documentation release

## v0.1.2

- README added

## v0.1.1

- Initial BlockingPy release