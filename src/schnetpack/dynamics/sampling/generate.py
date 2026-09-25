"""High-level generation entry point ("model in, structures out")."""

__all__ = ["generate"]


def generate(*args, **kwargs):
    """
    Generate structures from a trained model. Not implemented yet; assemble a
    :class:`~schnetpack.dynamics.sampling.sampler.Sampler` directly.
    """
    raise NotImplementedError(
        "schnetpack.dynamics.sampling.generate() lands with M1.3 (CLI + notebooks)."
    )
