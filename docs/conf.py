"""Sphinx configuration for the SchNetPack documentation.

Built by Read the Docs (see ``readthedocs.yaml``), which installs the package
with its ``doc`` extra before running Sphinx.  For a local build:

    pip install -e ".[doc]"
    make -C docs html
    # for clean build use: make clean && make html
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------

project = "SchNetPack"
author = (
    "Kristof T. Schütt, Stefan S. P. Hessmann, Niklas W. A. Gebauer, "
    "Jonas Lederer, Michael Gastegger"
)
copyright = f"2023, {author}"

# Read the version from the installed package so it never drifts from
# schnetpack.__version__, which pyproject.toml also builds from.
try:
    release = get_version("schnetpack")
except PackageNotFoundError:
    # Docs built without installing the package.
    release = "0.0.0"
# Short X.Y version, i.e. the release without patch level and pre-release tag.
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
]

templates_path = ["_templates"]
source_suffix = ".rst"  # nbsphinx registers ".ipynb" on its own
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
language = "en"
pygments_style = "sphinx"

# -- Extension configuration -------------------------------------------------

# Document both the class and the __init__ docstring.
autoclass_content = "both"
autosummary_generate = True

# Overrides that carry no docstring of their own (e.g. LightningModule hooks)
# stay blank instead of inheriting the base class' prose, which would document
# somebody else's API and pull in roles this build does not provide.
autodoc_inherit_docstrings = False

# Namespace auto-generated section labels by document, so that identically
# titled sections (e.g. "Requirements", "Summary") do not collide.
autosectionlabel_prefix_document = True

# The tutorials are shipped without stored outputs and are far too expensive to
# run during a docs build (they train models), so render them as-is.
nbsphinx_execute = "never"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "prev_next_buttons_location": "bottom",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
