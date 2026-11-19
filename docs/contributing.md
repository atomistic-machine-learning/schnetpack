# Contributing

Contributions are welcome as pull requests on
[GitHub](https://github.com/atomistic-machine-learning/schnetpack). Please open an
issue first for larger changes.

## Development setup

```bash
git clone https://github.com/atomistic-machine-learning/schnetpack.git
cd schnetpack
pip install -e ".[dev,test,doc]"
pre-commit install
```

pre-commit runs ruff (lint and format) and a few file checks on every commit; CI
runs the same hooks on every pull request.

## Tests

```bash
python -m pytest tests -q
```

The default selection skips tests marked `slow`, which includes every test that
compiles a model. Run them with `-m slow`.

## Docstrings

Docstrings follow the Google style (`Args:`, `Returns:`) and are written in
reStructuredText: link to other objects with roles such as
``:class:`~schnetpack.nn.Dense` `` and write equations with `:math:` or
`.. math::`. Type hints go in the signature; the API pages list them with the
parameters.

A new public class or function appears in the API reference only once its name is
listed in the matching page in `docs/api/`.

## Documentation

The documentation lives in `docs/`:

| Kind | Format |
| --- | --- |
| API reference (`docs/api/`) | reStructuredText (`.rst`) |
| Tutorials and guides with code | jupytext notebooks (`.py`) |
| Every other page | MyST Markdown (`.md`) |

Build it locally with:

```bash
make -C docs html                           # notebooks without outputs
NB_EXECUTION_MODE=force make -C docs html   # notebooks executed, as on Read the Docs
```

The HTML ends up in `docs/_build/html`. CI builds the docs on every pull request
with warnings as errors.

### Notebooks

Notebooks are stored as [jupytext](https://jupytext.readthedocs.io) `py:percent`
files, so they diff and merge like normal code and never carry outputs. Read the
Docs executes them on every build, which is where the outputs on the pages come
from.

To work on one interactively, open the `.py` in JupyterLab (with jupytext installed:
*right-click → Open With → Notebook*), or create a paired notebook:

```bash
jupytext --to notebook docs/tutorials/data.py
```

Jupyter then writes your edits back to the `.py`. The paired `.ipynb` holds the
outputs; it is gitignored, and a pre-commit hook rejects any `.ipynb`, so outputs
never reach the repository.

You can also edit the `.ipynb` in an editor that does not know about jupytext, such as
VS Code. The `jupytext-sync` pre-commit hook writes every paired notebook in
`docs/tutorials` and `docs/guides` back into its `.py` before a commit. If it rewrote a
`.py`, the commit fails: `git add` the `.py` and commit again. A notebook you created
from scratch gets a new `.py` that you have to `git add` yourself.

A notebook has to run on a CPU in a few minutes, downloads included. One that cannot
opts out of execution with a header at the top of its `.py`:

```python
# ---
# jupyter:
#   mystnb:
#     execution_mode: "off"
# ---
```
