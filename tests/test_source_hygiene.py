"""Whole-source checks that no single unit test would catch."""

import ast
import pathlib
import warnings

import schnetpack

SRC = pathlib.Path(schnetpack.__file__).parent


def test_no_syntax_warnings_in_source():
    """No invalid escape sequences anywhere in the package.

    ``"\\_"`` in a non-raw string is a SyntaxWarning today and becomes a
    SyntaxError in a future Python. A ``filterwarnings`` entry alone cannot
    guard this: SyntaxWarning is emitted at bytecode-compile time, so it fires
    only on the first uncached compile and stays silent once ``__pycache__`` is
    warm. Re-parsing every file from source makes the check deterministic.
    """
    failures = []
    for path in sorted(SRC.rglob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            try:
                ast.parse(path.read_text(), filename=str(path))
            except (SyntaxError, SyntaxWarning) as e:
                failures.append(f"{path.relative_to(SRC)}: {e}")

    assert not failures, "invalid escape sequences found:\n" + "\n".join(failures)
