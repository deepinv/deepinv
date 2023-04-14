# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'deepinverse'
copyright = '2023, DeepInv'
author = 'Julian Tachella, Matthieu Terris, Samuel Hurault and Dongdong Chen'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
intersphinx_mapping = {'python': ('https://deepinv.github.io/deepinv/', None)}


# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {
        "macros": {
            "forw": [r'{A\left({#1}\right)}', 1],
            "noise": [r'{N\left({#1}\right)}', 1],
            "inverse": [r'{R\left({#1}\right)}', 1],
            "inversef": [r'{R\left({#1},{#2}\right)}', 2],
            "reg": [r'{g\left({#1}\right)}', 1],
            "sensor": [r'{\eta\left({#1}\right)}', 1],
            "datafid": [r'{f\left({#1},{#2}\right)}', 2],
            "denoiser": [r'{D\left({#1},{#2}\right)}', 2],
            "xset": r'\mathcal{X}',
            "yset": r'\mathcal{Y}',
            "group": r'\mathcal{G}',
            "metric": [r'{d\left({#1},{#2}\right)}', 2],
            "loss": [r'{\mathcal\left({#1}\right)}', 1],
            }
        }
    }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = 'figures/logo.ico'
html_logo = "figures/deepinv_logolarge.png"
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

