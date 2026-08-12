"""Sphinx configuration for the SchNetPack documentation.

Built by Read the Docs (see ``readthedocs.yaml``), which installs the package
with its ``doc`` extra before running Sphinx.  For a local build:

    pip install -e ".[doc]"
    make -C docs html
    # for clean build use: make clean && make html
"""

import shutil
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

# -- Notebook sources --------------------------------------------------------

# The tutorials and how-tos live in examples/, but they cannot be pulled in with
# a symlink: nbsphinx links the figures an executed notebook produces relative to
# the notebook's real path, which for a symlinked directory points outside the
# source directory, and Sphinx then drops every one of them ("image file not
# readable"). So copy them in for the duration of the build. The copies are
# gitignored; examples/ stays the single source of truth.
_HERE = Path(__file__).parent.resolve()
_EXAMPLES = _HERE.parent / "examples"
for _name in ("tutorials", "howtos", "trained_models"):
    _dest = _HERE / _name
    if _dest.is_symlink():
        # Left over from a checkout that predates the copying.
        _dest.unlink()
    elif _dest.is_dir():
        shutil.rmtree(_dest)
    shutil.copytree(
        _EXAMPLES / _name,
        _dest,
        # Opening a tutorial in Jupyter leaves a paired .ipynb next to the .py
        # (see examples/jupytext.toml); both would claim the same docname.
        ignore=shutil.ignore_patterns("*.ipynb", ".ipynb_checkpoints", "__pycache__"),
    )

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
source_suffix = ".rst"  # nbsphinx registers the notebook formats on its own
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # nbsphinx_custom_formats below registers ".py" as a source suffix, and Sphinx
    # does not exclude its own config file, so without this it tries to render
    # conf.py as a document.
    "conf.py",
    # Same for any notebook that ends up here anyway: it would collide with the
    # .py of the same name.
    "**/*.ipynb",
    # Model checkpoints for howto_ensemble_calculation, not documents.
    "trained_models",
]
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

# The tutorials and how-tos are stored as jupytext py:percent files rather than
# .ipynb (see examples/jupytext.toml), which is why they carry no outputs at all.
# nbsphinx reads them through jupytext.
nbsphinx_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}

# opt out individually with an ``nbsphinx: execute: never``
# header in their .py file.
nbsphinx_execute = "auto"

nbsphinx_kernel_name = "python3"

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
