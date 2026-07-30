"""
Noising as a preprocessing transform.

This is where the tensor-level core meets SchNetPack's data pipeline.
:class:`Diffuse` runs the training half of a generative model — the process's
:meth:`~schnetpack.generative.processes.Process.perturb` plus
the parametrization's target — inside the dataloader, and writes the result
into the batch dict for an ordinary supervised loss to pick up.

It is deliberately :class:`~schnetpack.generative.losses.MatchingLoss` minus
the model call and the MSE. The two overlap because SchNetPack splits what
MatchingLoss fuses: noising belongs in a dataloader worker (parallel, per
structure, off the training thread), while the loss belongs in the task. Use
this one in a datamodule and pair it with a plain
:class:`~schnetpack.objectives.ModelOutput`; use MatchingLoss when you are
driving raw tensors and want the whole objective in one call.

Every axis stays swappable: the parametrization decides the label, and the
process decides the path, the endpoint distribution and the pairing. The
pair is validated at construction — use the same two objects here and in the
:class:`~schnetpack.generative.sampler.Sampler`.

Unlike the rest of the subpackage this module reaches into
``schnetpack.transform`` and ``schnetpack.properties``, since a transform is by
definition a statement about batch dicts. Nothing here imports Lightning.
"""

from typing import Callable, Optional, Sequence

import torch

from schnetpack import properties
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process
from schnetpack.transform.base import Transform

__all__ = ["Diffuse"]


class Diffuse(Transform):
    """
    Noise one property of a structure and expose the training target.

    Runs per structure, before collation, so exactly one time is drawn per
    structure and broadcast along the diffused property's leading axis. For
    positions that axis is atoms, which is what makes the tensor-level core
    apply unchanged: it only ever asked for a per-sample time, and here the
    samples are atoms.

    What it writes:

    - ``diffuse_property`` — overwritten with x_t, so the model and the
      neighbor list see the noised structure;
    - ``label_key`` — the parametrization's target, ready for an ordinary
      supervised :class:`~schnetpack.objectives.ModelOutput`;
    - ``time_key`` — the time, per element, for conditioning the head;
    - ``structure_time_key`` — the same time, once per structure, for a head
      that *predicts* the time.

    The time is written at both granularities on purpose. Collation
    concatenates along the leading axis, so a ``(n_atoms,)`` tensor arrives
    per-atom and a ``(1,)`` tensor arrives per-structure. A head that predicts
    one value per structure regressed against a per-atom target does not fail —
    MSE broadcasts ``(n_structures,)`` against ``(n_atoms,)`` and silently
    optimizes the wrong thing.

    It *does* tell the coupling which rows are interchangeable. The tensor-level
    core sees one anonymous sample axis; only here is it known that those rows
    are atoms, which molecule each belongs to and which element it is. So
    ``group_keys`` is read off the batch and handed to
    :meth:`~schnetpack.generative.processes.Process.perturb`, and a re-pairing
    coupling keeps its permutation inside one molecule and one element instead
    of trading endpoints across the whole batch.

    Two things this deliberately does *not* do:

    - **It does not center the structure.** Compose
      :class:`~schnetpack.transform.SubtractCenterOfGeometry` (or the
      center-of-mass variant) before it if your process lives in the zero-COM
      subspace.
    - **It does not decide how the noise is drawn.** That is the process's
      job — its prior. For molecules, translation-invariant networks cannot
      predict a center-of-mass displacement, so the noise must be drawn in
      the same zero-COM subspace as the data;
      :class:`~schnetpack.generative.priors.GaussianPrior` does that by
      default (``centered=True``). Express any other endpoint law as a
      :class:`~schnetpack.generative.priors.Prior` on the process, not by
      editing this class. This transform's part is only to hand the prior the
      batch as context, so a centered draw is centered per molecule rather
      than across the whole batch.

    Order matters in the transform list: put any neighbor list *after* this one,
    or it will be built on the clean structure and be wrong for x_t.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        t_sampler: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
        diffuse_property: str = properties.R,
        label_key: str = "label",
        time_key: str = "t",
        structure_time_key: Optional[str] = "t_structure",
        original_key: Optional[str] = None,
        group_keys: Optional[Sequence[str]] = (properties.idx_m, properties.Z),
    ):
        """
        Args:
            process: forward process that draws and places the endpoints
            parametrization: decides the label
            t_sampler: draws times, mapping (n, device) -> (n,); the same hook
                shape as :class:`~schnetpack.generative.losses.MatchingLoss`,
                so a sampler can be shared. Defaults to the process's own
                :meth:`~schnetpack.generative.processes.Process.sample_t`
                — uniform on [t_min, t_max], stopping short of t = 0 because
                the score target diverges there.
            diffuse_property: property to noise; overwritten with x_t
            label_key: key to write the training target to
            time_key: key for the per-element time, for conditioning
            structure_time_key: key for the per-structure time; None to skip
            original_key: key to keep the clean property under; None to skip
            group_keys: batch entries labelling which rows a re-pairing
                coupling may exchange endpoints between — by default the
                molecule index and the atomic number, so an atom trades only
                with atoms of its own element in its own molecule. Keys absent
                from the batch, or not one label per diffused row, are skipped;
                ``None`` or ``()`` leaves the assignment unrestricted. Ignored
                by couplings that do not re-pair.
        """
        super().__init__()
        parametrization.validate(process)
        self.process = process
        self.parametrization = parametrization
        self.t_sampler = t_sampler
        self.diffuse_property = diffuse_property
        self.label_key = label_key
        self.time_key = time_key
        self.structure_time_key = structure_time_key
        self.original_key = original_key
        self.group_keys = tuple(group_keys or ())

    def _groups(self, inputs, n: int) -> Optional[torch.Tensor]:
        """Stack the available group labels into one (n, k) tensor, or None."""
        columns = [
            inputs[key]
            for key in self.group_keys
            if key in inputs
            and inputs[key].ndim == 1
            and inputs[key].shape[0] == n  # per-structure properties have none
        ]
        return torch.stack(columns, dim=-1) if columns else None

    def forward(self, inputs):
        x0 = inputs[self.diffuse_property]

        # one time per structure, broadcast along the property's leading axis
        sample_t = self.t_sampler if self.t_sampler is not None else self.process.sample_t
        t = sample_t(1, x0.device).to(x0.dtype)
        t_elements = t.repeat(x0.shape[0])

        # the batch goes to the prior as context: a prior that must respect the
        # layout (centering per molecule) reads idx_m out of it, and one that
        # does not ignores it. Running per structure there is no idx_m, and the
        # whole leading axis is the one molecule anyway.
        x_t, x0, x1, t_elements, eps = self.process.perturb(
            x0,
            t=t_elements,
            context=inputs,
            groups=self._groups(inputs, x0.shape[0]),
        )

        inputs[self.diffuse_property] = x_t
        inputs[self.label_key] = self.parametrization.target(
            self.process, x0, x1, t_elements, eps
        )
        inputs[self.time_key] = t_elements
        if self.structure_time_key is not None:
            inputs[self.structure_time_key] = t
        if self.original_key is not None:
            inputs[self.original_key] = x0
        return inputs
