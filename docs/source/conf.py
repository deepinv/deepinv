# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# This is necessary for now but should not be in future version of sphinx_gallery
# as a simple list of paths will be enough.
from sphinx_gallery.sorting import ExplicitOrder, _SortKey, ExampleTitleSortKey
from sphinx_gallery.directives import ImageSg
import sys
import os
from sphinx.util import logging
import doctest
from importlib.metadata import metadata as importlib_metadata

logger = logging.getLogger(__name__)

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, basedir)

from deepinv.utils.plotting import set_default_plot_fontsize

set_default_plot_fontsize(12)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

metadata = importlib_metadata("deepinv")
project = str(metadata["Name"])
copyright = "deepinverse contributors 2025"
author = str(metadata["Author"])
release = str(metadata["Version"])

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
    "sphinx_sitemap",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_foot_reference_style = "foot"
copybutton_exclude = ".linenos, .gp"
bibtex_tooltips = True

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "python": ("https://docs.python.org/3.9/", None),
    "deepinv": ("https://deepinv.github.io/deepinv/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# for python3 type hints
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# to handle functions as default input arguments
autodoc_preserve_defaults = True
# Warn about broken links
nitpicky = True
# Create link to the API in the auto examples
autodoc_inherit_docstrings = False
# For bibtex
bibtex_footbibliography_backrefs = True
# for sitemap
html_baseurl = "https://deepinv.github.io/deepinv/"
html_extra_path = ["robots.txt"]
# Include reStructuredText sources
html_copy_source = True
# the default scheme makes for wrong urls so we specify it properly here
# For more details, see:
# https://sphinx-sitemap.readthedocs.io/en/v2.5.0/advanced-configuration.html
sitemap_url_scheme = "{link}"

####  userguide directive ###
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.addnodes import pending_xref

default_role = "code"  # default role for single backticks


class UserGuideMacro(Directive):
    required_arguments = 1  # The reference name (ref_name)
    has_content = False

    def run(self):
        ref_name = self.arguments[0]

        # Create the paragraph node
        paragraph_node = nodes.paragraph()

        # Add "**User Guide**: refer to " text
        paragraph_node += nodes.strong(text="User Guide: ")
        paragraph_node += nodes.Text("refer to ")

        # Create a pending_xref node to resolve the title dynamically
        xref_node = pending_xref(
            "",  # No initial text
            refdomain="std",  # Standard domain (used for :ref:)
            reftype="ref",  # Reference type
            reftarget=ref_name,  # Target reference
            refexplicit=False,  # Let Sphinx insert the title automatically
        )
        xref_node += nodes.Text("")  # Placeholder; Sphinx replaces this with the title
        paragraph_node += xref_node

        # Add the final " for more information." text
        paragraph_node += nodes.Text(" for more information.")

        return [paragraph_node]


class TolerantImageSg(ImageSg):
    option_spec = ImageSg.option_spec.copy()
    option_spec["ignore_missing"] = lambda x: True if x.lower() == "true" else False

    def run(self):
        image_path = self.arguments[0]
        ignore_missing = self.options.get("ignore_missing", False)
        full_path = os.path.join(basedir, "docs", "source", image_path.strip("/"))
        if (not os.path.exists(full_path)) and ignore_missing:
            logger.info(f"Ignoring missing image: {full_path}")
            return []  # Return empty node list to skip rendering
        return super().run()


def process_docstring(app, what, name, obj, options, lines):
    # Check if there is a footcite in the docstring
    if any(":footcite:" in line for line in lines):
        # Add the References section if not already present
        if not any(":References:" in line for line in lines):
            lines.append("")
            lines.append("|sep|")
            lines.append("")
            lines.append(":References:")
            lines.append("")
            lines.append(".. footbibliography::")
            lines.append("")


# Prevent indexing of viewcode pages by adding a meta noindex tag
# See also: https://developers.google.com/search/docs/crawling-indexing/block-indexing
def _noindex_viewcode(app, pagename, templatename, context, doctree):
    if pagename.startswith("_modules/"):
        context["metatags"] = (
            context.get("metatags", "") + '\n<meta name="robots" content="noindex">\n'
        )


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring, priority=10)
    app.add_directive("userguide", UserGuideMacro)
    app.add_directive("image-sg-ignore", TolerantImageSg)
    app.connect("html-page-context", _noindex_viewcode)


# ---------- doctest configuration -----------------------------------------
# Add a IGNORE_RESULT option to skip some line output
# From: https://stackoverflow.com/a/69780437/2642845

IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker

doctest_global_setup = """
import torch
import numpy as np
try:
    import astra
except ImportError:
    astra = None
cuda_available = torch.cuda.is_available()
"""


#############################


def add_references_block_to_examples():
    print("🔧 add_references_block_to_examples() called")
    for root, _, files in os.walk("../../examples"):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(root, fname)

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip if already has a bibliography block
            if "footbibliography" in content:
                continue

            # Add References block if footcite appears
            if ":footcite:" in content:
                references_block = (
                    "\n# %%\n" "# :References:\n" "#\n" "# .. footbibliography::\n"
                )
                content += references_block

                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)


add_references_block_to_examples()

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
add_module_names = True  # include the module path in the function name

from sphinx_gallery import gen_rst

gen_rst.EXAMPLE_HEADER = """
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "{0}"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        New to DeepInverse? Get started with the basics with the
        :ref:`5 minute quickstart tutorial <sphx_glr_auto_examples_basics_demo_quickstart.py>`.{2}

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_{1}:

"""

examples_order = {
    "basics": [
        "demo_quickstart.py",
        "demo_pretrained_model.py",
        "demo_custom_optim.py",
        "demo_custom_dataset.py",
        "demo_custom_physics.py",
    ],
    "models": [
        "demo_foundation_model.py",
        "demo_training.py",
        "demo_denoiser_tour.py",
    ],
    "physics": [
        "demo_physics_tour.py",
        "demo_blur_tour.py",
        "demo_mri_tour.py",
    ],
}


class MySortKey(_SortKey):
    """
    If section is listed in examples_order dict keys above, then sort
    examples by this custom order set by examples_order[section].
    If not, then sort by example titles.
    """

    def __call__(self, filename):
        section, example = os.path.normpath(os.path.join(self.src_dir, filename)).split(
            os.sep
        )[-2:]
        if section in examples_order:
            try:
                return examples_order[section].index(example)
            except ValueError:
                return len(examples_order[section]) + 1
        else:
            return ExampleTitleSortKey(self.src_dir)(filename)


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
            "../../examples/models",
            "../../examples/physics",
            "../../examples/optimization",
            "../../examples/plug-and-play",
            "../../examples/sampling",
            "../../examples/unfolded",
            "../../examples/self-supervised-learning",
            "../../examples/adversarial-learning",
            "../../examples/external-libraries",
        ]
    ),
    "within_subsection_order": MySortKey,
    "first_notebook_cell": (
        "# 🚀 To get started, install DeepInverse by creating a new cell and running `%pip install deepinv`\n"
    ),
}

# Custom sort key above throws new warning in Sphinx 7.3.0, so ignore this. See https://github.com/sphinx-doc/sphinx/issues/12300
suppress_warnings = ["config.cache"]

# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {
        "equationNumbers": {"autoNumber": "AMS", "useLabelIds": True},
        "macros": {
            "forw": [r"{A\left({#1}\right)}", 1],
            "noise": [r"{N\left({#1}\right)}", 1],
            "inverse": [r"{R\left({#1}\right)}", 1],
            "inversef": [r"{R\left({#1},{#2}\right)}", 2],
            "inversename": r"R",
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
html_css_files = ["custom.css"]
html_sidebars = {  # pages with no sidebar
    "contributing": [],
    "finding_help": [],
    "community": [],
}
html_theme_options = {
    "logo": {
        "image_light": "figures/deepinv_logolarge.png",
        "image_dark": "figures/logo_large_dark.png",
    },
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
    "analytics": {"google_analytics_id": "G-NSEKFKYSGR"},
}


# Separator substition : Writing |sep| in the rst file will display a horizontal line.
rst_prolog = """
.. |sep| raw:: html

   <hr />

"""

napoleon_custom_sections = [
    ("Reference", "params_style"),  # Sphinx ≥ 3.5
    # ("Reference", "Parameters"),   # fallback syntax for very old Sphinx (<3.5)
]

nitpick_ignore = [
    # This one generates a warning for some reason.
    ("py:class", "torchvision.transforms.InterpolationMode"),
]
