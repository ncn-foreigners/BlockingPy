# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os, sys
DOCS = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(DOCS, ".."))
sys.path.insert(0, os.path.join(ROOT, "packages", "blockingpy-core"))
sys.path.insert(0, os.path.join(ROOT, "packages", "blockingpy"))
sys.path.insert(0, os.path.join(ROOT, "packages", "blockingpy-gpu"))


project = "BlockingPy"
copyright = "2025, Tymoteusz Strojny and Maciej Beręsewicz"
author = "Tymoteusz Strojny, Maciej Beręsewicz"
release = "0.2.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": True, "inherited-members": True}

autodoc_typehints = "none"
autodoc_mock_imports = [
    "faiss", "faiss_cpu", "faiss_gpu",
    "mlpack", "hnswlib", "pynndescent", "voyager", "annoy",
    "igraph", "model2vec", "torch", "cupy", "sklearn", "nltk", "pooch",
]
autosummary_mock_imports = list(autodoc_mock_imports)



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

myst_enable_extensions = [
    "colon_fence",  # Enables ::: fence for directives
    "deflist",  # Enables definition lists
    "dollarmath",  # Enables dollar $ math syntax
    "fieldlist",  # Enables field lists
    "html_admonition",  # Enables HTML-style admonitions
    "html_image",  # Enables HTML-style images
    "replacements",  # Enables text replacements
    "smartquotes",  # Enables smart quotes
    "strikethrough",  # Enables strikethrough
    "tasklist",  # Enables task lists
]
