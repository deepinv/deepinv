# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# This is necessary for now but should not be in future version of sphinx_gallery
# as a simple list of paths will be enough.
from sphinx_gallery.sorting import ExplicitOrder

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "deepinverse"
copyright = "2024, deepinverse contributors"
author = (
    "Julian Tachella, Matthieu Terris, Samuel Hurault, Dongdong Chen and Andrew Wang"
)
release = "0.2"

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
    "sphinx_design",
    "sphinxcontrib.googleanalytics",
]
copybutton_exclude = ".linenos, .gp"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/2.0/", None),
    "python": ("https://docs.python.org/3.9/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
add_module_names = True # include the module path in the function name


sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/"],
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/demo_",
    "run_stale_examples": False,
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
    "nested_sections": True,
    "subsection_order": ExplicitOrder(
        [
            "../../examples/basics",
            "../../examples/optimization",
            "../../examples/plug-and-play",
            "../../examples/sampling",
            "../../examples/unfolded",
            "../../examples/patch-priors",
            "../../examples/self-supervised-learning",
            "../../examples/adversarial-learning",
            "../../examples/advanced",
        ]
    ),
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

html_theme = "pydata_sphinx_theme"
html_favicon = "figures/logo.ico"
html_static_path = ["_static"]
html_logo = "figures/deepinv_logolarge.png"
html_theme_options = {
    "secondary_sidebar_items": {
        "**": [
            "page-toc",
            "sourcelink",
            # Sphinx-Gallery-specific sidebar components
            # https://sphinx-gallery.github.io/stable/advanced.html#using-sphinx-gallery-sidebar-components
            "sg_download_links",
            "sg_launcher_links",
        ],
    },
}

html_js_files = ["js/custom.js"]

googleanalytics_id = "G-NSEKFKYSGR"

# Separator substition : Writing |sep| in the rst file will display a horizontal line.
rst_prolog = """
.. |sep| raw:: html

   <hr />
"""
