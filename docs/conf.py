"""Sphinx configuration for the SchNetPack documentation.

Read the Docs builds it (see ``readthedocs.yaml``) after installing the package with
its ``doc`` extra, and executes the notebooks while doing so. A local build:

    pip install -e ".[doc]"
    make -C docs html                            # notebooks shown without outputs
    NB_EXECUTION_MODE=force make -C docs html    # notebooks executed, as on RTD
    # for a clean build: make -C docs clean html

Pages are MyST Markdown (``.md``), the API reference is reStructuredText (``.rst``),
and the tutorials and guides with code are jupytext ``py:percent`` notebooks
(``.py``, see ``jupytext.toml``). No ``.ipynb`` is ever committed.
"""

import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

import schnetpack

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE / "_ext"))

# -- Project information -----------------------------------------------------

project = "SchNetPack"
author = "the SchNetPack authors"
copyright = f"2018–{date.today().year}, {author}"

# schnetpack.__version__ is what pyproject.toml builds from. Read it from the
# package rather than from installed metadata, which an editable install only
# refreshes on reinstall. autodoc imports the package anyway.
release = schnetpack.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",  # Markdown pages and notebooks
    "sphinx_design",  # cards and grids on the landing page
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",  # diagrams in the generative concept pages
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    # The notebooks are the .py files. A paired .ipynb that Jupyter leaves next
    # to one would claim the same page.
    "**/*.ipynb",
    # .py is a notebook suffix here (nb_custom_formats), so the Python files
    # that are not notebooks have to be excluded.
    "conf.py",
    "_ext",
]
language = "en"

# -- Notebooks ---------------------------------------------------------------

# Read the jupytext py:percent files as notebooks (see _ext/spk_nbreader.py).
nb_custom_formats = {".py": "spk_nbreader.reads"}

# Read the Docs executes the notebooks on every build; so does the `docs` job in
# .github/workflows/ci.yml. Local builds do not, unless NB_EXECUTION_MODE is set
# ("force" or "cache"). A notebook opts out with a jupytext header
# `mystnb: execution_mode: "off"`.
nb_execution_mode = os.environ.get(
    "NB_EXECUTION_MODE", "force" if os.environ.get("READTHEDOCS") else "off"
)
nb_execution_timeout = 600  # seconds per cell
nb_execution_raise_on_error = True
# Notebooks too slow for a CPU build, by path relative to docs/.
nb_execution_excludepatterns = []
if importlib.util.find_spec("mace") is None:
    # downloads MACE-OFF, which needs the `mace` extra
    nb_execution_excludepatterns.append("guides/batchwise_relaxations.py")

# Warnings that a notebook prints to stderr (with the paths of the build
# machine) stay off the page.
nb_output_stderr = "remove"

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# -- API reference -----------------------------------------------------------

# Document both the class and the __init__ docstring.
autoclass_content = "both"
autosummary_generate = True

# Overrides that carry no docstring of their own (e.g. LightningModule hooks)
# stay blank instead of inheriting the base class' prose, which would document
# somebody else's API and pull in roles this build does not provide.
autodoc_inherit_docstrings = False

# Short signatures; the types are listed with the parameters.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

# Copy code without prompts ($, >>>, ...).
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "SchNetPack"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
# Red Hat fonts and the colours of the SchNetPack paper figure
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/atomistic-machine-learning/schnetpack",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
        "text": "SchNetPack",
    },
    "announcement": (
        "This site covers SchNetPack 3. "
        "<a href='https://schnetpack.readthedocs.io/en/v2.3.x/'>"
        "Documentation for SchNetPack 2.x</a>"
    ),
}
