"""
Noising as a preprocessing transform.

This is where the tensor-level core meets SchNetPack's data pipeline.
:class:`Diffuse` runs the training half of a generative model — draw an endpoint
pair, sample a time, place the structure on the path, build the regression
target — inside the dataloader, and writes the result into the batch dict for an
ordinary supervised loss to pick up.

It is deliberately :class:`~schnetpack.generative.losses.MatchingLoss` minus the
model call and the MSE. The two overlap because SchNetPack splits what
MatchingLoss fuses: noising belongs in a dataloader worker (parallel, per
structure, off the training thread), while the loss belongs in the task. Use
this one in a datamodule and pair it with a plain
:class:`~schnetpack.objectives.ModelOutput`; use MatchingLoss when you are
driving raw tensors and want the whole objective in one call.

Every axis stays swappable: the parametrization decides the label and supplies
the path, the coupling decides how the second endpoint is drawn, and the time
sampler decides where along the path you land.

Unlike the rest of the subpackage this module reaches into
``schnetpack.transform`` and ``schnetpack.properties``, since a transform is by
definition a statement about batch dicts. Nothing here imports Lightning.
"""

from typing import Callable, Optional

import torch

from schnetpack import properties
from schnetpack.generative.couplings import Coupling, IndependentCoupling
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.paths import Path
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

    Two things this deliberately does *not* do:

    - **It does not center the structure.** Compose
      :class:`~schnetpack.transform.SubtractCenterOfGeometry` (or the
      center-of-mass variant) before it if your process lives in the zero-COM
      subspace.
    - **It does not decide how the noise is drawn.** That is the coupling's
      job. For molecules, translation-invariant networks cannot predict a
      center-of-mass displacement, so the noise must be projected into the same
      zero-COM subspace as the data — express that as a
      :class:`~schnetpack.generative.couplings.Coupling`, not by editing this
      class.

    Order matters in the transform list: put any neighbor list *after* this one,
    or it will be built on the clean structure and be wrong for x_t.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        parametrization: Parametrization,
        coupling: Optional[Coupling] = None,
        t_sampler: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
        diffuse_property: str = properties.R,
        label_key: str = "label",
        time_key: str = "t",
        structure_time_key: Optional[str] = "t_structure",
        original_key: Optional[str] = None,
    ):
        """
        Args:
            parametrization: decides the label and supplies the path
            coupling: how the second endpoint is drawn (default: independent
                standard normal). Constrained noise belongs here.
            t_sampler: draws times, mapping (n, device) -> (n,); the same hook
                shape as :class:`~schnetpack.generative.losses.MatchingLoss`,
                so a sampler can be shared. Default is uniform on
                [path.t_min, path.t_max] — it stops short of t = 0 because the
                score target diverges there.
            diffuse_property: property to noise; overwritten with x_t
            label_key: key to write the training target to
            time_key: key for the per-element time, for conditioning
            structure_time_key: key for the per-structure time; None to skip
            original_key: key to keep the clean property under; None to skip
        """
        super().__init__()
        self.parametrization = parametrization
        self.coupling = coupling if coupling is not None else IndependentCoupling()
        self.t_sampler = t_sampler if t_sampler is not None else self._uniform_t
        self.diffuse_property = diffuse_property
        self.label_key = label_key
        self.time_key = time_key
        self.structure_time_key = structure_time_key
        self.original_key = original_key

    @property
    def path(self) -> Path:
        """The path being diffused along, via the parametrization."""
        return self.parametrization.path

    def _uniform_t(self, n: int, device: Optional[torch.device]) -> torch.Tensor:
        span = self.path.t_max - self.path.t_min
        return self.path.t_min + span * torch.rand(n, device=device)

    def forward(self, inputs):
        x0 = inputs[self.diffuse_property]
        x0, x1 = self.coupling.sample(x0)

        # one time per structure, broadcast along the property's leading axis
        t = self.t_sampler(1, x0.device).to(x0.dtype)
        t_elements = t.repeat(x0.shape[0])

        inputs[self.diffuse_property] = self.path.interpolate(x0, x1, t_elements)
        inputs[self.label_key] = self.parametrization.target(x0, x1, t_elements)
        inputs[self.time_key] = t_elements
        if self.structure_time_key is not None:
            inputs[self.structure_time_key] = t
        if self.original_key is not None:
            inputs[self.original_key] = x0
        return inputs
