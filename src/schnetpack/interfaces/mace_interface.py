"""MACE-OFF compiled to TorchScript, for :class:`~schnetpack.train.MaceTeacher`.

A developer tool: it needs ``mace-torch``, which SchNetPack does not depend on::

    pip install "schnetpack[mace]"

:func:`compile_mace_off` writes ``MACE-OFF23_<size>_mace-<version>.pt`` to
:data:`CACHE_DIR`. The archive itself loads without ``mace-torch``.
"""

import contextlib
import os

import torch

__all__ = ["CACHE_DIR", "MACE_OFF_SIZES", "compile_mace_off"]

#: where :func:`compile_mace_off` writes by default; ignored by git
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MACE_OFF_SIZES = ("small", "medium", "large")

#: set by ``import mace`` so that e3nn 0.4.4 and MACE's pickled models load on
#: torch >= 2.6; left set, it would turn off ``weights_only`` for every later
#: ``torch.load`` in the process
_NO_WEIGHTS_ONLY = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def compile_mace_off(size: str = "medium", cache_dir: str = CACHE_DIR) -> str:
    """Compile MACE-OFF23 to a TorchScript archive and return its path.

    MACE downloads the weights on first use. The model is compiled in
    float64; an archive already in ``cache_dir`` is reused.

    Args:
        size: ``"small"``, ``"medium"`` or ``"large"``.
        cache_dir: directory the archive is written to.
    """
    if size not in MACE_OFF_SIZES:
        raise ValueError(
            f"unknown MACE-OFF size {size!r}, expected one of {MACE_OFF_SIZES}"
        )

    with _mace_environment():
        try:
            import mace
        except ImportError as err:
            raise ImportError(
                "mace-torch is needed to compile a MACE model; install it with "
                "pip install 'schnetpack[mace]'"
            ) from err

        path = os.path.join(cache_dir, f"MACE-OFF23_{size}_mace-{mace.__version__}.pt")
        if not os.path.exists(path):
            os.makedirs(cache_dir, exist_ok=True)
            _compile(size, path)
    return path


def _compile(size: str, path: str) -> None:
    from e3nn.util import jit
    from mace.calculators import mace_off

    calc = mace_off(model=size, default_dtype="float64", device="cpu")
    # written under another name first, so that an interrupted run is not taken
    # for a cached archive
    partial = path + ".part"
    torch.jit.save(jit.compile(calc.models[0]), partial)
    os.replace(partial, path)


@contextlib.contextmanager
def _mace_environment():
    """Unsafe ``torch.load`` while MACE loads its models, undone afterwards:
    ``import mace`` sets it, but on its first import only."""
    previous = os.environ.get(_NO_WEIGHTS_ONLY)
    os.environ[_NO_WEIGHTS_ONLY] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_NO_WEIGHTS_ONLY, None)
        else:
            os.environ[_NO_WEIGHTS_ONLY] = previous
