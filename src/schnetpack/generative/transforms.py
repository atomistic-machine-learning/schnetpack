"""
Noising as a preprocessing transform: the training half of a generative model
inside the SchNetPack data pipeline.

:class:`Diffuse` runs :meth:`~schnetpack.generative.processes.Process.perturb`
and the parametrization's target per structure in the dataloader and writes
the result into the batch dict, so training is an ordinary supervised loss.
It is :class:`~schnetpack.generative.losses.MatchingLoss` minus the model
call and the MSE. Details: ``docs_new/training.md``.
"""

from collections.abc import Callable, Sequence

import torch

from schnetpack import properties
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process
from schnetpack.transform.base import Transform

__all__ = ["Diffuse"]


class Diffuse(Transform):
    """
    Noise one property of a structure and expose the training target.

    Runs per structure, before collation: one time is drawn per structure and
    broadcast along the property's leading axis (atoms, for positions). Writes
    ``diffuse_property`` (overwritten with x_t), ``label_key`` (the target),
    ``time_key`` (the time per element) and ``structure_time_key`` (the time
    once per structure). The molecule index and atomic numbers are handed to
    the coupling as ``groups`` so a re-pairing stays within one element.

    It does not center the structure (compose
    :class:`~schnetpack.transform.SubtractCenterOfGeometry` before it) and
    does not decide how noise is drawn (that is the process's prior). Put any
    neighbor list *after* this transform so it is built on x_t.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        t_sampler: Callable[[int, torch.device], torch.Tensor] | None = None,
        diffuse_property: str = properties.R,
        label_key: str = "label",
        time_key: str = properties.t,
        structure_time_key: str | None = "t_structure",
        original_key: str | None = None,
        group_keys: Sequence[str] | None = (properties.idx_m, properties.Z),
    ):
        """
        Args:
            process: forward process that draws and places the endpoints
            parametrization: decides the label
            t_sampler: draws times, mapping (n, device) -> (n,); the same
                hook as :class:`~schnetpack.generative.losses.MatchingLoss`
                (default: the process's own
                :meth:`~schnetpack.generative.processes.Process.sample_t`)
            diffuse_property: property to noise; overwritten with x_t
            label_key: key to write the training target to
            time_key: key for the per-element time, for conditioning
            structure_time_key: key for the per-structure time; None to skip
            original_key: key to keep the clean property under; None to skip
            group_keys: batch entries labelling which rows a re-pairing
                coupling may exchange endpoints between. Keys absent from the
                batch, or not one label per diffused row, are skipped; None
                or () leaves the assignment unrestricted.
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

    def _groups(self, inputs, n: int) -> torch.Tensor | None:
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
        sample_t = (
            self.t_sampler if self.t_sampler is not None else self.process.sample_t
        )
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
