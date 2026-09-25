from .lr_scheduler import *


_MOVED_TO_LIGHTNING = (
    "ModelCheckpoint",
    "PredictionWriter",
    "ExponentialMovingAverage",
)


def __getattr__(name):
    if name in _MOVED_TO_LIGHTNING:
        import warnings

        warnings.warn(
            f"`schnetpack.train.{name}` moved to `schnetpack.lightning.{name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from schnetpack import lightning

        return getattr(lightning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
