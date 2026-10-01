"""
Hooks: callbacks around the steps of a
:class:`~schnetpack.dynamics.base.Dynamics` loop.

A hook runs before a step (changing what the model sees) or after it
(changing what the step produced), e.g. :class:`FreezeScaffold`, or only
observes.
"""

import torch

from schnetpack import properties

__all__ = ["Hook", "FreezeScaffold"]


class Hook:
    """
    Base class of the hooks; both methods default to identity.

    They receive the current batch, the step counter and the number of
    steps, and return the, possibly new, batch. Which keys a hook edits is
    its own configuration. ``step`` counts completed steps: ``before_step``
    sees the index of the step about to run, ``after_step`` the count
    including it. Return a new dict rather than editing in place.
    """

    def before_step(self, batch, step: int, n_steps: int):
        """Edit the batch before the step: what the model will see."""
        return batch

    def after_step(self, batch, step: int, n_steps: int):
        """Edit the batch after the step: what the step produced."""
        return batch


class FreezeScaffold(Hook):
    """
    Hold the atoms marked in ``batch[mask_key]`` at ``batch[reference_key]``.

    The scaffold rows of ``batch[key]`` are overwritten with the reference
    before and after every step, on any dynamics: the model always sees the
    clean scaffold and the run ends on it. Give the reference in the frame
    the model expects; overwriting rows breaks the centering of a centered
    prior draw.
    """

    def __init__(
        self,
        key: str = properties.R,
        mask_key: str = properties.fixed_atoms,
        reference_key: str = properties.R_reference,
    ):
        """
        Args:
            key: batch key of the frozen values, usually the key the dynamics
                moves
            mask_key: batch key of the boolean per-atom scaffold mask
            reference_key: batch key of the reference values, shaped like the
                frozen key; only the masked rows are read
        """
        self.key = key
        self.mask_key = mask_key
        self.reference_key = reference_key

    def _freeze(self, batch):
        x = batch[self.key]
        mask = batch[self.mask_key]
        if mask.shape != x.shape[:1]:
            raise ValueError(
                f"{self.mask_key!r} must hold one flag per row: shape "
                f"{tuple(x.shape[:1])}, got {tuple(mask.shape)}"
            )
        reference = batch[self.reference_key].to(dtype=x.dtype, device=x.device)
        if reference.shape != x.shape:
            raise ValueError(
                f"{self.reference_key!r} must be shaped like {self.key!r} "
                f"{tuple(x.shape)}, got {tuple(reference.shape)}"
            )
        x = torch.where(mask.view(-1, *[1] * (x.dim() - 1)), reference, x)
        return {**batch, self.key: x}

    def before_step(self, batch, step, n_steps):
        return self._freeze(batch)

    def after_step(self, batch, step, n_steps):
        return self._freeze(batch)
