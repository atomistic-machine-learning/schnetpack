"""
Inference on spk batches:

A :class:`Calculator` owns the device and dtype, the neighbor list, the
gradient policy and the model call. It never writes into the batch it is
given: the neighbor list and the model get a shallow copy, so the keys they
add never reach the driver's batch.

The plain :class:`Calculator` never touches units: the batch reaches the
model as the driver holds it. That is what a generative model needs — its
raw head has no single unit (x0 is a length, the noise has none, the score
is an inverse length) and its process lives in the training data's units,
so a sampler steps in the model's units. :class:`ForceFieldCalculator` is the
unit boundary of a force field: the driver works in eV and Angstrom, the
model in its own units.

:class:`EnsembleCalculator` is the same seam for an ensemble of force
fields: it reports the members' mean, which drives any driver exactly as a
single model's outputs would, and how far the members disagree.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from schnetpack import properties
from schnetpack.uncertainty import AbsoluteUncertainty, Uncertainty
from schnetpack.units import convert_units

__all__ = [
    "Calculator",
    "ForceFieldCalculator",
    "EnsembleCalculator",
    "NNEnsemble",
    "as_calculator",
]


class Calculator:
    """
    Run a model on spk batches: ``calculator(batch) -> outputs``.

    Stateless between calls apart from what the neighbor list caches;
    :meth:`reset` clears that between runs. No output caching by default. A
    driver that asks twice about the same structure — the
    :class:`~schnetpack.dynamics.relax.Relaxer`, whose convergence check and
    step rule both need the forces at the current positions — turns on
    ``cache_last``, and pays for one model call instead of two.
    """

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        neighbor_list: Callable[[dict], dict] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        enable_grad: bool = False,
        cache_last: bool = False,
    ):
        """
        Args:
            model: batch -> outputs, e.g. a
                :class:`~schnetpack.model.NeuralNetworkPotential`, or the
                path to one saved on disk.
            neighbor_list: batch -> batch, run on every call to rebuild the
                neighbor list of the current structures. Without one, the
                batch's own neighbor keys are used as they are, which is
                stale for a cutoff list once the structures move.
            device: device to run on; the model and every batch are moved
                there (default: leave both where they are)
            dtype: floating dtype to run in; the model and the floating
                tensors of every batch are cast to it (default: unchanged)
            enable_grad: run the model with autograd enabled, for models that
                differentiate their outputs (forces from an energy)
            cache_last: remember the outputs of the last call and return them
                again while the batch is unchanged.
        """
        if isinstance(model, str):
            from schnetpack.utils.compatibility import load_model

            model = load_model(model, device="cpu").to(torch.float64)
        if isinstance(model, nn.Module) and (device is not None or dtype is not None):
            model = model.to(device=device, dtype=dtype)
        self.model = model
        self.neighbor_list = neighbor_list
        self.device = device
        self.dtype = dtype
        self.enable_grad = enable_grad
        self.cache_last = cache_last
        self._cached_key = None
        self._cached_outputs = None

    def prepare(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """
        Move a batch to the calculator's device and dtype, once per run.

        Returns a new dict; non-tensor values pass through.
        """
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                dtype = (
                    self.dtype
                    if self.dtype is not None and value.is_floating_point()
                    else None
                )
                value = value.to(device=self.device, dtype=dtype)
            out[key] = value
        return out

    def reset(self) -> None:
        """Forget per-run caches (the neighbor list's, the last outputs)."""
        self._cached_key = None
        self._cached_outputs = None
        reset = getattr(self.neighbor_list, "reset", None)
        if reset is not None:
            reset()

    @staticmethod
    def _batch_key(batch: Mapping[str, Any]) -> tuple:
        # Holding the tensors themselves keeps them alive, so a freed tensor
        # can never be mistaken for the cached one by a recycled id.
        return tuple(
            (key, value, value._version)
            for key, value in batch.items()
            if torch.is_tensor(value)
        )

    def _cache_hit(self, key: tuple) -> bool:
        if self._cached_key is None or len(key) != len(self._cached_key):
            return False
        return all(
            k1 == k2 and v1 is v2 and n1 == n2
            for (k1, v1, n1), (k2, v2, n2) in zip(key, self._cached_key)
        )

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.cache_last:
            key = self._batch_key(batch)
            if self._cache_hit(key):
                return dict(self._cached_outputs)
        outputs = self._evaluate(batch)
        if self.cache_last:
            # detached: an attached output would keep its whole graph alive
            # until the next call
            self._cached_outputs = {
                k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()
            }
            self._cached_key = key
            return dict(self._cached_outputs)
        return outputs

    def _evaluate(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        inputs = dict(batch)
        if self.enable_grad and properties.R in inputs:
            # the model marks the positions as requiring grad to differentiate
            # by them; on a detached view that never reaches the driver's
            # tensor, which stays an ordinary value it can step
            inputs[properties.R] = inputs[properties.R].detach()
        if self.neighbor_list is not None:
            inputs = self.neighbor_list(inputs)
        with torch.set_grad_enabled(self.enable_grad):
            return self.model(inputs)


class ForceFieldCalculator(Calculator):
    """
    Run a force field on spk batches held in eV and Angstrom.

    The driver's batch is in Angstrom and the outputs come back in eV,
    eV/Angstrom and eV/Angstrom^3, whatever units the model works in. The
    positions and the cell are converted to the model's units on the copy the
    model is handed — before the neighbor list, whose cutoff is in the
    model's units — and the energy, forces and stress back from them. Any
    other output is returned as the model reported it. A model that does not
    return energy and forces is reported rather than run on.

    This is the calculator a :class:`~schnetpack.dynamics.relax.Relaxer`
    runs on. It runs with autograd enabled by default, for models that
    differentiate their energy. A generative driver refuses it: its raw head
    is not a force-field quantity, and would come back unconverted.
    """

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        energy_unit: str | float = "eV",
        position_unit: str | float = "Ang",
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        stress_key: str | None = None,
        enable_grad: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: batch -> outputs, or the path to a model saved on disk
            energy_unit: energy unit the model works in
            position_unit: length unit the model works in
            energy_key: model output holding the energy per structure
            force_key: model output holding the forces
            stress_key: model output holding the stress; None if the model
                predicts none
            enable_grad: run the model with autograd enabled

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        super().__init__(model, enable_grad=enable_grad, **kwargs)
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")
        self.output_units = {
            energy_key: self.energy_conversion,
            force_key: self.energy_conversion / self.position_conversion,
        }
        if stress_key is not None:
            self.output_units[stress_key] = (
                self.energy_conversion / self.position_conversion**3
            )

    def _evaluate(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.position_conversion != 1.0:
            batch = {
                **batch,
                **{
                    key: batch[key] / self.position_conversion
                    for key in (properties.R, properties.cell)
                    if key in batch
                },
            }
        outputs = super()._evaluate(batch)
        if self.energy_key not in outputs or self.force_key not in outputs:
            raise KeyError(
                f"a force field must return {self.energy_key!r} and "
                f"{self.force_key!r}; got {sorted(outputs)}"
            )
        return {
            key: value * self.output_units[key] if key in self.output_units else value
            for key, value in outputs.items()
        }


class NNEnsemble(nn.Module):
    """
    Several models evaluated together, reported as one prediction per model.

    An ensemble is itself a model: it takes a batch and returns a dictionary
    of predictions. What is different is that every entry carries a leading
    model axis, ``(n_models, ...)``, so the caller can take the mean, look at
    the spread, or both. Reducing that axis is left to the caller —
    :class:`EnsembleCalculator` averages it for the outputs that drive a
    loop, and hands the unreduced stack to an
    :class:`~schnetpack.uncertainty.Uncertainty`.
    """

    def __init__(self, models: nn.ModuleList, properties: str | list[str]):
        """
        Args:
            models: the ensemble members
            properties: names of the properties to collect from them
        """
        super().__init__()
        self.models = models
        if isinstance(properties, str):
            properties = [properties]
        self.properties = properties

    def setup(self, stage: str | None = None) -> None:
        for model in self.models:
            model.setup(stage)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        collected: dict[str, list[torch.Tensor]] = {p: [] for p in self.properties}
        for model in self.models:
            # a shallow copy per member, so the entries one model adds to the
            # batch (pairwise distances, say) are not seen by the next one;
            # the tensors themselves are shared
            predictions = model(dict(inputs))
            for prop in collected:
                if prop in predictions:
                    collected[prop].append(predictions[prop])
        return {
            prop: torch.stack(values) for prop, values in collected.items() if values
        }


class EnsembleCalculator(ForceFieldCalculator):
    """
    Run an ensemble of force fields on spk batches, reporting how much it
    disagrees.

    The outputs are the members' mean for every collected property, so any
    driver runs on them exactly as on a single model's. On top of that
    ``outputs[uncertainty_key]`` holds how far the members disagreed, one
    value per structure — which, after a relaxation, tells *which*
    structures went somewhere the models were not trained. Several
    uncertainty functions give a dictionary keyed by class name.

    Like any :class:`ForceFieldCalculator` it takes batches in Angstrom and
    reports energy, forces and stress in eV and Angstrom; the uncertainty is
    computed on the converted members, in the units
    :class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`
    reports it in, so a criterion tuned against one applies to the other.
    """

    def __init__(
        self,
        models: str | Sequence[str] | Sequence[nn.Module] | nn.ModuleList,
        properties: Sequence[str] = ("energy", "forces"),
        uncertainty_fn: Uncertainty | list[Uncertainty] | None = None,
        energy_unit: str | float = "eV",
        position_unit: str | float = "Ang",
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: str | None = None,
        uncertainty_key: str = "uncertainty",
        **kwargs,
    ):
        """
        Args:
            models: the ensemble members: a list of paths, a list of modules,
                a ``ModuleList``, or the path of a directory laid out as
                schnetpack training leaves it, ``<dir>/<run>/best_model``
            properties: model outputs to collect from the members
            uncertainty_fn: one :class:`~schnetpack.uncertainty.Uncertainty`
                or a list of them (default:
                :class:`~schnetpack.uncertainty.AbsoluteUncertainty`)
            energy_unit: energy unit the models work in
            position_unit: length unit the models work in
            energy_key: energy output
            force_key: force output
            stress_key: stress output; None if the models predict none
            uncertainty_key: output key the uncertainty is reported under

        Remaining keyword arguments are those of :class:`ForceFieldCalculator`.
        """
        members = nn.ModuleList(
            [
                member if isinstance(member, nn.Module) else self._load_member(member)
                for member in self._resolve(models)
            ]
        )
        super().__init__(
            NNEnsemble(members, list(properties)),
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            **kwargs,
        )

        if uncertainty_fn is None:
            uncertainty_fn = AbsoluteUncertainty(
                energy_key=energy_key, force_key=force_key, stress_key=stress_key or ""
            )
        if not isinstance(uncertainty_fn, list):
            uncertainty_fn = [uncertainty_fn]
        self.uncertainty_fn = uncertainty_fn
        self.uncertainty_key = uncertainty_key

    @staticmethod
    def _resolve(models) -> list:
        """A directory of trained models is expanded; anything else is a list."""
        if isinstance(models, str):
            return [
                os.path.join(models, run, "best_model")
                for run in sorted(os.listdir(models))
            ]
        return list(models)

    @staticmethod
    def _load_member(path: str) -> nn.Module:
        from schnetpack.utils.compatibility import load_model

        return load_model(path, device="cpu").to(torch.float64)

    def _evaluate(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        # the model axis is kept by the ensemble, and the unit factors
        # broadcast over it: the mean drives the loop, and the spread across
        # it is the uncertainty
        stacked = super()._evaluate(batch)
        outputs = {prop: value.mean(dim=0) for prop, value in stacked.items()}

        members = {prop: value.detach() for prop, value in stacked.items()}
        n_atoms = batch.get(properties.n_atoms)
        if len(self.uncertainty_fn) == 1:
            outputs[self.uncertainty_key] = self.uncertainty_fn[0](members, n_atoms)
        else:
            outputs[self.uncertainty_key] = {
                type(fn).__name__: fn(members, n_atoms) for fn in self.uncertainty_fn
            }
        return outputs


def as_calculator(calculator) -> Calculator:
    """Pass a :class:`Calculator` through; wrap a bare ``batch -> outputs`` callable."""
    if isinstance(calculator, Calculator):
        return calculator
    return Calculator(calculator)
