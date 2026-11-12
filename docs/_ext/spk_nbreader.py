"""Read a jupytext ``py:percent`` file as a notebook for myst_nb.

``jupytext.toml`` strips the kernel metadata from the files, but myst_nb needs it
to pick the syntax highlighting and the kernel that executes the notebook.
"""

import jupytext


def reads(text, **kwargs):
    nb = jupytext.reads(text, fmt="py:percent")
    nb.metadata.setdefault(
        "kernelspec",
        {"name": "python3", "display_name": "Python 3", "language": "python"},
    )
    return nb
