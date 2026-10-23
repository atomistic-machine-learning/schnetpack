"""
The two ways a dynamics driver consumes a model.

A calculator runs a model for one family of drivers:

- :class:`ForceCalculator` hands the time-free drivers a force at the
  current structures. Its ``kind`` says which: a *physical* force, the
  negative gradient of an energy, in eV/Angstrom; or a *pseudo* force, a
  displacement such as GPFF's 2 (x0 - x), in Angstrom, with no energy
  behind it. It adds its :mod:`~schnetpack.dynamics.guidance` terms to the
  forces, so a driver sees only the guided force.
- :class:`FieldCalculator` hands the
  :class:`~schnetpack.dynamics.sample.sampler.Sampler` the model's raw
  outputs at time t. Each update of the sampler converts its variable's head
  into the field it steps on and adds its own guidance.

Both share the plumbing of :class:`Calculator`: the device and dtype, the
neighbor list, the gradient policy, the optional cache of the last outputs,
and the rule that the driver's batch is never written to — the neighbor list
and the model get a shallow copy, so the keys they add never reach it.

:class:`EnsembleCalculator` is a :class:`ForceCalculator` for an ensemble of
force fields, run as one :class:`~schnetpack.model.NNEnsemble`: it reports
the models' mean, which drives any driver exactly as a single model's
outputs would, and how far the models disagree, one value per structure
for every :class:`~schnetpack.uncertainty.Uncertainty`.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import nn

from schnetpack import properties
from schnetpack.dynamics.guidance import Guidance, guidance_terms, weighted_sum
from schnetpack.generative.variables import STRUCTURE_TIME_KEY
from schnetpack.model.ensemble import NNEnsemble
from schnetpack.uncertainty import AbsoluteUncertainty, Uncertainty
from schnetpack.units import convert_units

__all__ = [
    "Calculator",
    "ForceCalculator",
    "FieldCalculator",
    "EnsembleCalculator",
]


def _load_model(path: str) -> nn.Module:
    """A model saved on disk, on the cpu and in float64."""
    from schnetpack.utils.compatibility import load_model

    return load_model(path, device="cpu").to(torch.float64)


class Calculator:
    """
    Run a model on spk batches: ``calculator(batch) -> outputs``.

    The shared base of :class:`ForceCalculator` and
    :class:`FieldCalculator`. It never touches units: the batch reaches
    the model as the driver holds it.

    Stateless between calls apart from what the neighbor list caches;
    :meth:`reset` clears that between runs. No output caching by default. A
    driver that asks twice about the same structure — a convergence check and
    a step rule that both need the forces at the current positions — turns on
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
            model = _load_model(model)
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


class ForceCalculator(Calculator):
    """
    Run a model that predicts a force on spk batches held in Angstrom.

    ``kind`` says what the force is:

    - ``"physical"``: the negative gradient of an energy (an MLFF). The
      outputs come back in eV, eV/Angstrom and eV/Angstrom^3, whatever units
      the model works in, and a model that does not return energy and forces
      is reported rather than run on. Runs with autograd enabled by default,
      for models that differentiate their energy.
    - ``"pseudo"``: a displacement toward the data with no energy behind it,
      such as GPFF's F = 2 (x0 - x). It is a length, so it comes back in
      Angstrom. The model sees the path time 0 under ``time_key``: a
      time-free driver has no noise level to report.

    :meth:`forces` adds the guidance terms — restraints such as
    :class:`~schnetpack.dynamics.guidance.HarmonicRestraint` — to the
    model's forces, each scaled by its ``weight``; the outputs of a plain
    call are the model's alone. On a physical force the weight is a plain
    factor. A pseudo-force is minus the gradient of the pseudo-energy
    ||x - x0||^2, in Angstrom^2, and a term of energy E enters it as w E: the
    weight converts eV to Angstrom^2, so it is in Angstrom^2/eV, and the
    weighted term w F is in Angstrom like the pseudo-force.

    The positions and the cell are converted to the model's units on the copy
    the model is handed — before the neighbor list, whose cutoff is in the
    model's units — and the energy, forces and stress back from them. Any
    other output is returned as the model reported it. With the default
    ``position_unit`` of Angstrom nothing is converted, which is also how a
    pseudo-force model trained on data in Angstrom runs.

    The drivers read ``kind`` for the defaults that depend on it: the L-BFGS
    starting curvature (70 eV/Angstrom^2 or 2) and the unit of the stop
    tolerance (eV/Angstrom or Angstrom).
    """

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        kind: Literal["physical", "pseudo"] = "physical",
        energy_unit: str | float = "eV",
        position_unit: str | float = "Ang",
        energy_key: str | None = properties.energy,
        force_key: str | None = None,
        stress_key: str | None = None,
        time_key: str = properties.t,
        enable_grad: bool | None = None,
        guidance: Sequence[Guidance] = (),
        **kwargs,
    ):
        """
        Args:
            model: batch -> outputs, or the path to a model saved on disk
            kind: ``"physical"`` for the gradient of an energy,
                ``"pseudo"`` for a displacement with no energy behind it
            energy_unit: energy unit the model works in; unused for a
                pseudo-force
            position_unit: length unit the model works in
            energy_key: model output holding the energy per structure;
                unused for a pseudo-force
            force_key: model output holding the force (default:
                ``properties.forces`` for a physical force, ``"prediction"``
                for a pseudo-force, the raw head of a generative model)
            stress_key: model output holding the stress; None if the model
                predicts none. Physical forces only.
            time_key: batch key the zero path time is written to for a
                pseudo-force model
            enable_grad: run the model with autograd enabled (default: for a
                physical force only)
            guidance: terms added to the model's forces by :meth:`forces`

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        if kind not in ("physical", "pseudo"):
            raise ValueError(f"kind must be 'physical' or 'pseudo', got {kind!r}")
        physical = kind == "physical"
        if not physical and stress_key is not None:
            raise ValueError("a pseudo-force has no energy, and so no stress")
        if enable_grad is None:
            enable_grad = physical
        super().__init__(model, enable_grad=enable_grad, **kwargs)
        self.kind = kind
        self.guidance = guidance_terms(guidance)
        self.energy_key = energy_key if physical else None
        self.force_key = force_key or (properties.forces if physical else "prediction")
        self.stress_key = stress_key
        self.time_key = time_key
        self.position_conversion = convert_units(position_unit, "Angstrom")
        if physical:
            self.energy_conversion = convert_units(energy_unit, "eV")
            self.output_units = {
                self.energy_key: self.energy_conversion,
                self.force_key: self.energy_conversion / self.position_conversion,
            }
            if stress_key is not None:
                self.output_units[stress_key] = (
                    self.energy_conversion / self.position_conversion**3
                )
        else:
            self.energy_conversion = None
            self.output_units = {self.force_key: self.position_conversion}

    @property
    def physical(self) -> bool:
        """Whether the force is the gradient of an energy."""
        return self.kind == "physical"

    def forces(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """
        The force at ``batch``, the model's plus the guidance's: eV/Angstrom
        if physical, Angstrom if pseudo.
        """
        # detached outputs from the cache: adding the terms below neither
        # reaches the cache nor keeps the model's graph alive
        forces = self(batch)[self.force_key]
        if self.guidance:
            forces = forces + weighted_sum(self.guidance, batch).to(forces)
        return forces

    def energy(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """The energy per structure at ``batch``, in eV. Physical forces only."""
        if not self.physical:
            raise TypeError("a pseudo-force has no energy behind it")
        return self(batch)[self.energy_key]

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
        if not self.physical:
            positions = batch[properties.R]
            batch = {
                **batch,
                self.time_key: positions.new_zeros(positions.shape[0]),
            }
        outputs = super()._evaluate(batch)
        required = (
            [self.energy_key, self.force_key] if self.physical else [self.force_key]
        )
        missing = [key for key in required if key not in outputs]
        if missing:
            raise KeyError(
                f"a {self.kind} force model must return {required}; "
                f"got {sorted(outputs)}"
            )
        return {
            key: value * self.output_units[key] if key in self.output_units else value
            for key, value in outputs.items()
        }


class FieldCalculator(Calculator):
    """
    One model call at time t, read by every update of a
    :class:`~schnetpack.dynamics.sample.sampler.Sampler`.

    Returns the model's outputs as they are, raw heads included: each update
    reads its variable's head and converts it into the field it steps on
    with the variable's parametrization. The time goes in twice, as
    :class:`~schnetpack.generative.transforms.Diffuse` writes it in
    training: per atom under ``time_key`` and per structure under
    ``structure_time_key``.
    """

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        time_key: str = properties.t,
        structure_time_key: str = STRUCTURE_TIME_KEY,
        **kwargs,
    ):
        """
        Args:
            model: batch -> outputs with the head of every variable the
                updates step, or the path to a model saved on disk
            time_key: batch key for the time per atom
            structure_time_key: batch key for the time per structure

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        super().__init__(model, **kwargs)
        self.time_key = time_key
        self.structure_time_key = structure_time_key

    def with_time(self, batch: Mapping[str, Any], t: torch.Tensor) -> dict[str, Any]:
        """
        A copy of ``batch`` with the time t per atom and per structure.

        Args:
            batch: structures with ``idx_m`` and ``n_atoms``
            t: the time, 0-dim
        """
        return {
            **batch,
            self.time_key: t.expand(batch[properties.idx_m].shape[0]),
            self.structure_time_key: t.expand(batch[properties.n_atoms].shape[0]),
        }


class EnsembleCalculator(ForceCalculator):
    """
    Run an ensemble of force fields on spk batches, reporting how much it
    disagrees.

    The outputs are the models' mean for every collected property, so any
    driver runs on them exactly as on a single model's. On top of that every
    :class:`~schnetpack.uncertainty.Uncertainty` adds one value per structure
    under its own key — which, after a relaxation, tells *which* structures
    went somewhere the models were not trained. A single uncertainty is
    reported under ``"uncertainty"``; a mapping ``{key: uncertainty}`` reports
    each under its key.

    A physical :class:`ForceCalculator`: it takes batches in Angstrom and
    reports energy, forces and stress in eV and Angstrom. The uncertainties
    are computed on the converted predictions, keyed by the names of
    :mod:`schnetpack.properties` whatever the models call their outputs, in
    the units
    :class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`
    reports them in, so a criterion tuned against one applies to the other.

    The models run as one :class:`~schnetpack.model.NNEnsemble` on the same
    batch, so they share its units and neighbor list.
    """

    def __init__(
        self,
        models: str | Sequence[str] | Sequence[nn.Module] | nn.ModuleList,
        uncertainty: Uncertainty | Mapping[str, Uncertainty] | None = None,
        energy_unit: str | float = "eV",
        position_unit: str | float = "Ang",
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        stress_key: str | None = None,
        extra_properties: Sequence[str] = (),
        **kwargs,
    ):
        """
        Args:
            models: the models of the ensemble: a list of paths, a list of modules,
                a ``ModuleList``, or the path of a directory laid out as
                schnetpack training leaves it, ``<dir>/<run>/best_model``
            uncertainty: one :class:`~schnetpack.uncertainty.Uncertainty`,
                reported under ``"uncertainty"``, or a mapping from output key
                to uncertainty (default:
                :class:`~schnetpack.uncertainty.AbsoluteUncertainty`)
            energy_unit: energy unit the models work in
            position_unit: length unit the models work in
            energy_key: energy output
            force_key: force output
            stress_key: stress output; None if the models predict none
            extra_properties: further outputs to average over the models,
                returned unconverted

        Remaining keyword arguments are ``guidance`` and those of
        :class:`Calculator`.
        """
        loaded = nn.ModuleList(
            [
                model if isinstance(model, nn.Module) else _load_model(model)
                for model in self._resolve(models)
            ]
        )
        collected = [energy_key, force_key]
        if stress_key is not None:
            collected.append(stress_key)
        collected += list(extra_properties)
        super().__init__(
            NNEnsemble(loaded, collected),
            kind="physical",
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            **kwargs,
        )

        if uncertainty is None:
            uncertainty = AbsoluteUncertainty()
        if not isinstance(uncertainty, Mapping):
            uncertainty = {"uncertainty": uncertainty}
        for key, fn in uncertainty.items():
            if not isinstance(fn, Uncertainty):
                raise TypeError(f"{key!r}: {type(fn).__name__} is not an Uncertainty")
            if key in collected:
                raise ValueError(
                    f"uncertainty key {key!r} would overwrite the averaged {key!r}"
                )
        self.uncertainty = dict(uncertainty)
        # the model's names for the properties an Uncertainty reads by their
        # canonical ones
        self._canonical = {energy_key: properties.energy, force_key: properties.forces}
        if stress_key is not None:
            self._canonical[stress_key] = properties.stress

    @staticmethod
    def _resolve(models) -> list:
        """A directory of trained models is expanded; anything else is a list."""
        if isinstance(models, str):
            return [
                os.path.join(models, run, "best_model")
                for run in sorted(os.listdir(models))
            ]
        return list(models)

    def _evaluate(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        # the model axis is kept by the ensemble, and the unit factors
        # broadcast over it: the mean drives the loop, and the spread across
        # it is the uncertainty
        stacked = super()._evaluate(batch)
        outputs = {prop: value.mean(dim=0) for prop, value in stacked.items()}

        predictions = {
            self._canonical[prop]: value.detach()
            for prop, value in stacked.items()
            if prop in self._canonical
        }
        n_atoms = batch.get(properties.n_atoms)
        for key, fn in self.uncertainty.items():
            outputs[key] = fn(predictions, n_atoms)
        return outputs
