"""High-level generation entry point ("model in, structures out")."""

__all__ = ["generate"]


def generate(*args, **kwargs):
    """
    Generate structures from a trained model: sample compositions and starting
    states from a prior, run a :class:`~schnetpack.generative.sampler.Sampler`
    batch-wise until the requested number of structures is produced, and
    convert the result to ASE Atoms.

    Lands with M1.3, together with ``cli.generate()`` and the ``spkgenerate``
    entry point.
    """
    raise NotImplementedError(
        "schnetpack.generative.generate() lands with M1.3 (CLI + notebooks)."
    )
