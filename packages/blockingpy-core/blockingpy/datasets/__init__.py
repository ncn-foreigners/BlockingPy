"""
Module containing example datasets for BlockingPy.

This module provides built-in datasets for testing, demonstration,
and benchmarking of blocking algorithms. The datasets include:

- Census-CIS Data: Census and CIS datasets for record linkage
- Deduplication Data (RLdata10000): Artificial patient records with introduced errors

Data Attribution:
----------------
Census-CIS Data:
    Created by Paula McLeod, Dick Heasman and Ian Forbes, ONS,
    for the ESSnet DI on-the-job training course, Southampton,
    25-28 January 2011
    https://wayback.archive-it.org/12090/20231221144450/https://cros-legacy.ec.europa.eu/content/job-training_en

Deduplication Data:
    Taken from "RecordLinkage" R package developed by Murat Sariyar
    and Andreas Borg. Package is licensed under GPL-3 license.
    https://cran.r-project.org/package=RecordLinkage

"""

from .base import load_census_cis_data, load_deduplication_data

__all__ = [
    "load_census_cis_data",
    "load_deduplication_data",
]
