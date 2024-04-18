# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "deepinverse"
copyright = "2024, DeepInv"
author = "Julian Tachella, Matthieu Terris, Samuel Hurault and Dongdong Chen"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",
    "sphinxemoji.sphinxemoji",
    "sphinx_copybutton",
]

copybutton_exclude = ".linenos, .gp"

intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://docs.pytorch.org/2.0/", None),
    "python": ("https://docs.python.org/3.4", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/"],
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/demo_",
    "run_stale_examples": True,
    "ignore_pattern": r"__init__\.py",
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("deepinv"),
    # objects to exclude from implicit backreferences. The default option
    # is an empty set, i.e. exclude nothing.
    "exclude_implicit_doc": {},
    "nested_sections": False,
}

# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {
        "equationNumbers": {"autoNumber": "AMS", "useLabelIds": True},
        "macros": {
            "forw": [r"{A\left({#1}\right)}", 1],
            "noise": [r"{N\left({#1}\right)}", 1],
            "inverse": [r"{R\left({#1}\right)}", 1],
            "inversef": [r"{R\left({#1},{#2}\right)}", 2],
            "reg": [r"{g_\sigma\left({#1}\right)}", 1],
            "regname": r"g_\sigma",
            "sensor": [r"{\eta\left({#1}\right)}", 1],
            "datafid": [r"{f\left({#1},{#2}\right)}", 2],
            "datafidname": r"f",
            "distance": [r"{d\left({#1},{#2}\right)}", 2],
            "distancename": r"d",
            "denoiser": [r"{\operatorname{D}_{{#2}}\left({#1}\right)}", 2],
            "denoisername": r"\operatorname{D}_{\sigma}",
            "xset": r"\mathcal{X}",
            "yset": r"\mathcal{Y}",
            "group": r"\mathcal{G}",
            "metric": [r"{d\left({#1},{#2}\right)}", 2],
            "loss": [r"{\mathcal\left({#1}\right)}", 1],
            "conj": [r"{\overline{#1}^{\top}}", 1],
        },
    }
}

math_numfig = True
numfig = True
numfig_secnum_depth = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_favicon = "figures/logo.ico"
html_logo = "figures/deepinv_logolarge.png"
html_theme_options = {
    "analytics_id": "G-NSEKFKYSGR",  # Provided by Google in your dashboard G-
    "analytics_anonymize_ip": False,
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Separator substition : Writing |sep| in the rst file will display a horizontal line.
rst_prolog = """
.. |sep| raw:: html

   <hr />
"""
