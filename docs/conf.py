# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AnotherSPDNet'
copyright = '2023, Ammar Mian, Florent Bouchard, Paul Chauchat'
author = 'Ammar Mian, Florent Bouchard, Paul Chauchat'
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = ['sphinx.ext.autodoc', "sphinx.ext.viewcode",
              'sphinx.ext.coverage','sphinx.ext.napoleon',
              "sphinx.ext.mathjax"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'description': 'A PyTorch implementation of SPDNet',
    'github_user': 'ammarmian',
    'github_repo': 'anotherspdnet',
    'sidebar_width': '255px',
    'github_button': False,
    'font_family': 'Helvetica',
}
