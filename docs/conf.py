# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'BlockingPy'
copyright = '2024, Tymoteusz Strojny'
author = 'Tymoteusz Strojny'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
    
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

myst_enable_extensions = [
    "colon_fence",    # Enables ::: fence for directives
    "deflist",        # Enables definition lists
    "dollarmath",     # Enables dollar $ math syntax
    "fieldlist",      # Enables field lists
    "html_admonition", # Enables HTML-style admonitions
    "html_image",     # Enables HTML-style images
    "replacements",   # Enables text replacements
    "smartquotes",    # Enables smart quotes
    "strikethrough", # Enables strikethrough
    "tasklist"       # Enables task lists
]
