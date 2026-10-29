"""
Noising as a preprocessing transform: the training half of a generative model
inside the SchNetPack data pipeline.

:class:`Diffuse` noises every :class:`~schnetpack.generative.variables.Variable`
of a structure at one time in the dataloader and writes each variable's x_t,
label and loss weight into the batch dict, so training is an ordinary
supervised loss per variable. Theory:
``docs_new/generation/new_variables.md``.
"""

from collections.abc import Callable, Sequence

import torch

from schnetpack import properties
from schnetpack.generative.variables import STRUCTURE_TIME_KEY, Variable
from schnetpack.transform.base import Transform

__all__ = ["Diffuse"]


class Diffuse(Transform):
    """
    Noise every variable of a structure at one shared time, in order.

    Runs per structure before collation, or on a collated batch (one with
    ``idx_m``, e.g. the padded batches of
    :class:`~schnetpack.lightning.DeviceAtomsDataModule`, where rows marked as padding
    get weight 0). One time is drawn per structure and
    each variable is noised at it, reading x0 through its view and writing
    x_t back; each variable sees the structure the ones before it left. Writes
    for every variable its x_t (into the structure), its label and its loss
    weight, and once ``time_key`` per atom and ``structure_time_key`` per
    structure. The molecule index and the
    atomic numbers are handed to the coupling of a per-atom variable as
    ``groups``, so a re-pairing stays within one element.

    It does not center the structure (compose
    :class:`~schnetpack.transform.SubtractCenterOfGeometry` before it) and
    does not decide how noise is drawn (that is each variable's prior). Put
    any neighbor list after this transform, so it is built on the noisy
    structure.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        variables: Sequence[Variable],
        t_sampler: Callable[[int, torch.device], torch.Tensor] | None = None,
        time_key: str = properties.t,
        structure_time_key: str | None = STRUCTURE_TIME_KEY,
        keep_clean: bool = False,
        group_keys: Sequence[str] | None = (properties.idx_m, properties.Z),
    ):
        """
        Args:
            variables: the variables, in the order they are noised
            t_sampler: draws times, mapping (n, device) -> (n,), e.g. a
                :class:`~schnetpack.generative.times.LogNormalSigmaTimes`
                (default: uniform on the range every variable's schedule
                covers)
            time_key: key for the time per atom, for conditioning
            structure_time_key: key for the time per structure; None to skip
            keep_clean: also keep each variable's x0 under its
                :attr:`~schnetpack.generative.variables.Variable.clean_key`
            group_keys: batch entries labelling which rows a re-pairing
                coupling may exchange endpoints between. Keys absent from the
                batch, or not one label per row of the variable, are skipped;
                None or () leaves the assignment unrestricted.
        """
        super().__init__()
        self.variables = list(variables)
        if not self.variables:
            raise ValueError("Diffuse needs at least one variable")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"variable names must be unique, got {names}")
        schedules = [v.noising.schedule for v in self.variables]
        self.t_min = max(s.t_min for s in schedules)
        self.t_max = min(s.t_max for s in schedules)
        if self.t_min >= self.t_max:
            raise ValueError(
                f"the variables' schedules share no time range: "
                f"[{self.t_min}, {self.t_max}]"
            )
        self.t_sampler = t_sampler
        self.time_key = time_key
        self.structure_time_key = structure_time_key
        self.keep_clean = keep_clean
        self.group_keys = tuple(group_keys or ())

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Times for n structures: the ``t_sampler``, or uniform on the shared range."""
        if self.t_sampler is not None:
            return self.t_sampler(n, device)
        return self.t_min + (self.t_max - self.t_min) * torch.rand(n, device=device)

    def _groups(self, inputs, n: int) -> torch.Tensor | None:
        """Stack the available group labels into one (n, k) tensor, or None."""
        columns = [
            inputs[key]
            for key in self.group_keys
            if key in inputs and inputs[key].ndim == 1 and inputs[key].shape[0] == n
        ]
        return torch.stack(columns, dim=-1) if columns else None

    def forward(self, inputs):
        # One structure (the data pipeline) or a collated batch (idx_m present),
        # e.g. the padded batches of DeviceAtomsDataModule: one time per structure,
        # spread to the rows through idx_m, and padding rows weighted 0.
        idx_m = inputs.get(properties.idx_m)
        first = self.variables[0].read(inputs)
        n_structures = 1 if idx_m is None else inputs[properties.n_atoms].shape[0]
        t = self.sample_t(n_structures, first.device)
        floating = None
        for v in self.variables:
            x0 = v.read(inputs)
            dtype = x0.dtype if x0.is_floating_point() else t.dtype
            if floating is None and x0.is_floating_point():
                floating = dtype
            if idx_m is None:
                t_rows = t.to(dtype).repeat(x0.shape[0])
            else:
                t_rows = (t[idx_m] if v.per == "atom" else t).to(dtype)
            # the batch goes to the prior, which may read the layout: running
            # per structure there is no idx_m, and the whole leading axis is
            # the one structure
            x_t, x0, x1, t_rows = v.noising.perturb(
                x0,
                t=t_rows,
                batch=inputs,
                groups=self._groups(inputs, x0.shape[0]) if v.per == "atom" else None,
            )
            schedule = v.noising.schedule
            inputs = v.write(inputs, x_t)
            inputs[v.label_key] = v.parametrization.target(schedule, x0, x1, t_rows)
            weight = (
                v.weight(schedule, t_rows, x_t)
                if v.weight is not None
                else torch.ones_like(t_rows)
            )
            mask = inputs.get(
                properties.atom_mask if v.per == "atom" else properties.structure_mask
            )
            inputs[v.weight_key] = weight if mask is None else weight * mask.to(weight)
            if self.keep_clean:
                inputs[v.clean_key] = x0

        t = t.to(floating or t.dtype)
        if idx_m is not None:
            inputs[self.time_key] = t[idx_m]
        elif properties.Z in inputs:
            inputs[self.time_key] = t.repeat(inputs[properties.Z].shape[0])
        elif any(v.per == "atom" for v in self.variables):
            atom_rows = next(v for v in self.variables if v.per == "atom")
            inputs[self.time_key] = t.repeat(atom_rows.read(inputs).shape[0])
        if self.structure_time_key is not None:
            inputs[self.structure_time_key] = t
        return inputs
