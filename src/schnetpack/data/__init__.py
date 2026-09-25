from .atoms import *
from .loader import *
from .stats import *
from .splitting import *
from .sampler import *
from .provider import *


def __getattr__(name):
    if name == "AtomsDataModule":
        import warnings

        warnings.warn(
            "`schnetpack.data.AtomsDataModule` moved to "
            "`schnetpack.lightning.AtomsDataModule`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from schnetpack.lightning import AtomsDataModule

        return AtomsDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
