"""
Inference on spk batches:

A :class:`Calculator` owns everything between "here is a batch" and "here are
the model outputs": moving the batch to the model's device and dtype, the
neighbor list, the gradient policy and the model call itself. Drivers own
none of it; they hand the calculator a batch and read the outputs.

The calculator never writes into the batch it is given. The neighbor list
and the model both write into their inputs, so they get a shallow copy, and
the keys they add (neighbor lists, distance vectors, outputs) never reach the
driver's batch. That is what keeps a moving structure free of stale derived
keys: they only ever exist on the copy built for one call.

:class:`EnsembleCalculator` is the same seam for an ensemble of models: it
reports the members' mean, which drives any driver exactly as a single
model's outputs would, and how far the members disagree.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from schnetpack import properties
from schnetpack.uncertainty import AbsoluteUncertainty, Uncertainty
from schnetpack.units import convert_units

__all__ = ["Calculator", "EnsembleCalculator", "NNEnsemble", "as_calculator"]


class Calculator:
    """
    Run a model on spk batches: ``calculator(batch) -> outputs``.

    Stateless between calls apart from whatever the neighbor list caches
    (a skin-based list keeps the lists it built); :meth:`reset` clears that
    between runs. No output caching by default: an integrator such as Heun
    evaluates different structures within one step, so a cache would never
    hit. A driver that asks twice about the same structure — the
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
                batch's own neighbor keys are used as they are — right for a
                static (e.g. fully connected) list, stale for a cutoff list
                once the structures move.
            device: device to run on; the model and every batch are moved
                there (default: leave both where they are)
            dtype: floating dtype to run in; the model and the floating
                tensors of every batch are cast to it (default: unchanged)
            enable_grad: run the model with autograd enabled. Needed for
                models that differentiate their outputs (forces from an
                energy); generative heads run faster without it.
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


class EnsembleCalculator(Calculator):
    """
    Run an ensemble on spk batches, reporting how much it disagrees.

    The outputs are the members' mean for every collected property, so any
    driver runs on them exactly as on a single model's. On top of that
    ``outputs[uncertainty_key]`` holds how far the members disagreed, one
    value per structure — which, after a relaxation or a sampling run, tells
    *which* structures went somewhere the models were not trained. Several
    uncertainty functions give a dictionary keyed by class name.

    The means stay in the model's units, like any calculator's outputs. The
    uncertainty is reported in eV and Angstrom, the units
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
            energy_key: energy output, converted for the uncertainty
            force_key: force output, converted for the uncertainty
            stress_key: stress output, converted for the uncertainty; None
                if the models predict none
            uncertainty_key: output key the uncertainty is reported under

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        members = nn.ModuleList(
            [
                member if isinstance(member, nn.Module) else self._load_member(member)
                for member in self._resolve(models)
            ]
        )
        super().__init__(NNEnsemble(members, list(properties)), **kwargs)

        energy = convert_units(energy_unit, "eV")
        length = convert_units(position_unit, "Angstrom")
        self.uncertainty_units = {energy_key: energy, force_key: energy / length}
        if stress_key is not None:
            self.uncertainty_units[stress_key] = energy / length**3

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
        # the model axis is kept by the ensemble: the mean drives the loop, and
        # the spread across it is the uncertainty
        stacked = super()._evaluate(batch)
        outputs = {prop: value.mean(dim=0) for prop, value in stacked.items()}

        converted = {
            prop: value.detach() * self.uncertainty_units.get(prop, 1.0)
            for prop, value in stacked.items()
        }
        n_atoms = batch.get(properties.n_atoms)
        if len(self.uncertainty_fn) == 1:
            outputs[self.uncertainty_key] = self.uncertainty_fn[0](converted, n_atoms)
        else:
            outputs[self.uncertainty_key] = {
                type(fn).__name__: fn(converted, n_atoms) for fn in self.uncertainty_fn
            }
        return outputs


def as_calculator(calculator) -> Calculator:
    """Pass a :class:`Calculator` through; wrap a bare ``batch -> outputs`` callable."""
    if isinstance(calculator, Calculator):
        return calculator
    return Calculator(calculator)
