import os
import sys
from importlib import import_module
from configparser import ConfigParser
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nDspec'
copyright = '2023, Matteo Lucchini, Phil Uttley'
author = 'Matteo Lucchini, Phil Uttley'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx', 'sphinx.ext.autodoc'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'

html_theme_options = {
    #'nosidebar': True,
    'navigation_with_keys': True,
    'description': 'Jupyter Notebooks + Sphinx',
    'github_banner': True,
    'github_button': True,
    'github_repo': 'nbsphinx',
    'github_type': 'star',
    'github_user': 'spatialaudio',
    'page_width': '1095px',
    #'show_relbars': True,
}
html_sidebars = {
    '**': [
        'about.html',
        #'globaltoc.html',
        #'localtoc.html',
        'navigation.html',
        'searchbox.html',
        #'sourcelink.html',
    ]
}

#import stuff for the api docs
#note that the line below is messy with astropy affiliated packages so it will
#be a problem in the long run
sys.path.insert(0, os.path.abspath('../ndspec/'))
conf = ConfigParser()
conf.read([os.path.join(os.path.dirname(__file__), "..", "setup.cfg")])
setup_cfg = dict(conf.items("metadata"))

autodoc_mock_imports = ['bs4', 'requests']

html_static_path = ['_static']
