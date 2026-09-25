"""
State-level constraints: edits of the iterate between the steps of a
:class:`~schnetpack.dynamics.base.Dynamics` loop.

A constraint hooks in before a step (it changes what the model sees) or after
it (it changes what the step produced), or both. Constraints act on the
*state*; changing the *field* the step follows (restraint forces, guidance)
is a different seam and does not live here.

On a time-aware sampler the iterate at time t must stay on the noise
manifold at t: an edit that moves it off (projecting a bond length on a noisy
x_t, say) feeds the model inputs it never saw in training. :class:`Scaffold`
re-noises for that reason. On a time-free loop any edit is fine.
"""

import torch

from schnetpack import properties

__all__ = ["StateConstraint", "AnnealedNoise", "Scaffold"]


class StateConstraint:
    """
    Base class of state-level constraints; both hooks default to identity.

    Hooks receive the current batch, the step counter, the number of steps
    and the running :class:`~schnetpack.dynamics.base.Dynamics` (for its
    moved key, time key, process, prior and ``time_free`` flag), and return
    the — possibly new — batch. ``step`` counts completed steps: a
    before-step hook sees the index of the step about to run, an after-step
    hook the count including it (``step == n_steps`` after the last one).
    Return a new dict (``{**batch, key: value}``) rather than editing the
    batch in place.
    """

    def before_step(self, batch, step: int, n_steps: int, dynamics):
        """Edit the batch before the step: what the model will see."""
        return batch

    def after_step(self, batch, step: int, n_steps: int, dynamics):
        """Edit the batch after the step: what the step produced."""
        return batch


class AnnealedNoise(StateConstraint):
    """
    GPFF's decaying noise injection before each step:

        x <- x + lambda (1 - k/N) z,   z ~ N(0, I),

    for step k = 1..N, on the driver's moved key. The last step injects
    nothing. lambda is in data units (Angstrom, for positions); it is what
    buys the stochastic variant of direct denoising its sample diversity.
    """

    def __init__(self, stochastic_lambda: float = 1.0):
        """
        Args:
            stochastic_lambda: scale of the injected noise, in data units
        """
        self.stochastic_lambda = stochastic_lambda

    def before_step(self, batch, step, n_steps, dynamics):
        noise_scale = self.stochastic_lambda * (1.0 - (step + 1) / n_steps)
        if noise_scale <= 0.0:
            return batch
        x = batch[dynamics.key]
        return {**batch, dynamics.key: x + noise_scale * torch.randn_like(x)}


class Scaffold(StateConstraint):
    """
    Hold the atoms marked in ``batch[mask_key]`` at ``batch[reference_key]``.

    Both are per-atom batch keys, so they collate with the structures and a
    batch may carry a different scaffold per molecule; the mask is the same
    convention the relaxers use for fixed atoms.

    Before every step the scaffold rows of the moved key are set to where
    the scaffold belongs at the iterate's noise level, so the model always
    sees the scaffold, and the others are generated around it:

    - time-free dynamics (``dynamics.time_free``, GPFF's direct denoising):
      overwritten with the reference positions.
    - time-aware dynamics (the sampler): re-noised to the current time
      through the process, ``a(t) x_ref + b(t) x1`` with x1 a fresh prior
      draw — the RePaint-style inpainting step, which keeps x_t on the noise
      manifold.

    After the final step the scaffold rows are overwritten with the reference
    positions, so the output holds the scaffold exactly.

    Overwriting rows breaks the zero-center-of-geometry a centered
    :class:`~schnetpack.generative.priors.GaussianPrior` guarantees; give the
    reference in the frame the model expects (typically centered on the
    whole structure it will be part of).
    """

    def __init__(
        self,
        mask_key: str = properties.fixed_atoms,
        reference_key: str = properties.R_reference,
    ):
        """
        Args:
            mask_key: batch key of the boolean per-atom scaffold mask
            reference_key: batch key of the reference values, shaped like the
                moved key; only the masked rows are read
        """
        self.mask_key = mask_key
        self.reference_key = reference_key

    def _mask_and_reference(self, batch, x):
        mask = batch[self.mask_key].to(dtype=torch.bool, device=x.device)
        if mask.shape != x.shape[:1]:
            raise ValueError(
                f"{self.mask_key!r} must hold one flag per row: shape "
                f"{tuple(x.shape[:1])}, got {tuple(mask.shape)}"
            )
        reference = batch[self.reference_key].to(dtype=x.dtype, device=x.device)
        if reference.shape != x.shape:
            raise ValueError(
                f"{self.reference_key!r} must be shaped like the moved key "
                f"{tuple(x.shape)}, got {tuple(reference.shape)}"
            )
        return mask, reference

    @staticmethod
    def _overwrite(batch, dynamics, mask, values):
        x = torch.where(
            mask.view(-1, *[1] * (values.dim() - 1)), values, batch[dynamics.key]
        )
        return {**batch, dynamics.key: x}

    def before_step(self, batch, step, n_steps, dynamics):
        x = batch[dynamics.key]
        mask, reference = self._mask_and_reference(batch, x)
        if dynamics.time_free:
            return self._overwrite(batch, dynamics, mask, reference)
        # A full-size prior draw lets the prior see the layout it expects in
        # the batch; only the scaffold rows of the noised reference are kept.
        x1 = dynamics.prior.sample(
            x.shape, dtype=x.dtype, device=x.device, context=batch
        )
        t = batch[dynamics.time_key]
        return self._overwrite(
            batch, dynamics, mask, dynamics.process.interpolate(reference, x1, t)
        )

    def after_step(self, batch, step, n_steps, dynamics):
        if step < n_steps:
            return batch
        x = batch[dynamics.key]
        mask, reference = self._mask_and_reference(batch, x)
        return self._overwrite(batch, dynamics, mask, reference)
