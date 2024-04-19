import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))

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
    'nbsphinx',
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

autodoc_mock_imports = ['bs4', 'requests']

html_static_path = ['_static']
