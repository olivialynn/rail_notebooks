# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

from importlib.metadata import version

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath('../src/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RAIL Notebooks"
copyright = "2023, LF"
author = "LF"
release = version("rail_notebooks")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.mathjax", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

templates_path = []
exclude_patterns = ['_build', '**.ipynb_checkpoints']

master_doc = "index"  # This assumes that sphinx-build is called from the root directory
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
add_module_names = False # Remove namespaces from class/method signatures

"""
autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_ignore = ["*/__main__.py", "*/_version.py"]
autoapi_add_toc_tree_entry = False
autoapi_member_order = "bysource"
"""

# -- Grab the demonstrations file from the main RAIL repo --------------------
from urllib.request import urlretrieve

urlretrieve (
    "https://raw.githubusercontent.com/LSSTDESC/rail/main/docs/source/demonstrations.rst",
    "demonstrations.rst"
)

# -- CSS ---------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/notebooks.css',
]
