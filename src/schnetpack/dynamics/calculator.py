"""
The two ways a dynamics driver consumes a model.

A calculator turns a model into the field one family of drivers steps on:

- :class:`ForceCalculator` hands the time-free drivers a force at the
  current structures. Its ``kind`` says which: a *physical* force, the
  negative gradient of an energy, in eV/Angstrom; or a *pseudo* force, a
  displacement such as GPFF's 2 (x0 - x), in Angstrom, with no energy
  behind it.
- :class:`GenerativeCalculator` hands the time-indexed drivers the score,
  the denoised sample x0 or the probability-flow velocity at (x, t). It owns
  the model together with the process and the parametrization it was trained
  under, validated once, so the drivers read the process from here.

Both share the plumbing of :class:`Calculator`: the device and dtype, the
neighbor list, the gradient policy, the optional cache of the last outputs,
and the rule that the driver's batch is never written to — the neighbor list
and the model get a shallow copy, so the keys they add never reach it. Both
also take :mod:`~schnetpack.dynamics.guidance` terms, which they add to the
field they return: to the forces, or to the score (and through it to x0 and
the velocity). A driver sees only the guided field.

:class:`EnsembleCalculator` is a :class:`ForceCalculator` for an ensemble of
force fields: it reports the members' mean, which drives any driver exactly
as a single model's outputs would, and how far the members disagree.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import nn

from schnetpack import properties
from schnetpack.dynamics.guidance import Guidance
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process, expand_t
from schnetpack.uncertainty import AbsoluteUncertainty, Uncertainty
from schnetpack.units import convert_units

__all__ = [
    "Calculator",
    "ForceCalculator",
    "GenerativeCalculator",
    "EnsembleCalculator",
    "NNEnsemble",
]


class Calculator:
    """
    Run a model on spk batches: ``calculator(batch) -> outputs``.

    The shared base of :class:`ForceCalculator` and
    :class:`GenerativeCalculator`. It never touches units: the batch reaches
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
        guidance: Sequence[Guidance] = (),
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
            guidance: terms added to the field this calculator returns; the
                subclass says how (:meth:`guidance_field`)
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
        self.guidance = list(guidance)
        for term in self.guidance:
            if not isinstance(term, Guidance):
                raise TypeError(f"{type(term).__name__} is not a Guidance")

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

    def guidance_field(
        self, batch: Mapping[str, Any], positions: torch.Tensor
    ) -> torch.Tensor | None:
        """
        The sum of the guidance terms at ``positions``, each scaled by its
        ``weight``.

        Every term is called on a copy of ``batch`` with ``positions`` under
        ``properties.R``.

        Args:
            batch: current batch
            positions: positions in Angstrom to evaluate at

        Returns:
            The field ``(n_atoms, 3)`` in eV/Angstrom, or None without
            guidance.
        """
        if not self.guidance:
            return None
        inputs = {**batch, properties.R: positions}
        field = torch.zeros_like(positions)
        for term in self.guidance:
            field = field + term.weight * term(inputs)
        return field

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
        field = self.guidance_field(batch, batch[properties.R])
        if field is not None:
            forces = forces + field.to(forces)
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


class GenerativeCalculator(Calculator):
    """
    Run a generative model on spk batches and read its head as a field.

    Owns the model together with the process and the parametrization it was
    trained under; the pair is validated at construction, so it cannot
    diverge between the calculator and the driver, which reads
    :attr:`process` from here. :meth:`score`, :meth:`x0` and
    :meth:`velocity` evaluate the model at (x, t) and convert its raw head,
    ``outputs[output_key]``; the driver asks for whichever its step needs.

    The batch is in the model's units: the process's prior and schedule are
    fixed in the training data's units, and the raw head has no single unit
    of its own (x0 is a length, the noise has none, the score is an inverse
    length), so nothing is converted. ``position_unit`` names that length
    unit for the guidance alone.

    Guidance — restraints such as
    :class:`~schnetpack.dynamics.guidance.HarmonicRestraint`, or terms
    kT ∇ log p — guides the score: the sum of the weighted terms
    F = sum_i w_i F_i is added to it, score + F. A term's ``weight`` w_i is
    its 1/kT, in 1/eV, so a restraint of weight w tilts the sampled density
    by exp(-w E). The terms see the batch with the path time under
    ``time_key``, so a term may depend on t. The other fields follow from the
    guided score through the process's (f, g) chart, the velocity as
    v - 1/2 g^2 F and x0 by Tweedie's formula as x0 + sigma^2 / a F, so
    guidance requires the chart, which is acquired at construction. The
    guidance is evaluated at the noisy iterate x_t, not at a clean estimate:
    an approximation that is exact only as t -> t_min. :meth:`raw` stays the
    model's own head.
    """

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        process: Process,
        parametrization: Parametrization,
        key: str = properties.R,
        output_key: str = "prediction",
        time_key: str = properties.t,
        position_unit: str | float = "Ang",
        **kwargs,
    ):
        """
        Args:
            model: batch -> outputs, or the path to a model saved on disk
            process: forward process the model was trained on
            parametrization: contract the model was trained under
            key: batch key the fields are taken with respect to
            output_key: model output holding the raw head
            time_key: batch key the path time is written to, one value per
                row of ``key`` (the key
                :class:`~schnetpack.generative.transforms.Diffuse` wrote in
                training)
            position_unit: length unit of ``key`` — the process's space,
                i.e. the training data's. Used only to evaluate the guidance
                in Angstrom.

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        parametrization.validate(process)
        super().__init__(model, **kwargs)
        self.process = process
        self.parametrization = parametrization
        self.key = key
        self.output_key = output_key
        self.time_key = time_key
        self.position_conversion = convert_units(position_unit, "Angstrom")
        # guidance reaches the velocity and x0 through the chart: acquire it
        # now, so a configuration without one fails here, not mid-run
        self.sde = process.sde() if self.guidance else None

    @staticmethod
    def _rows(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``t`` as one value per row of ``x``."""
        return t.expand(x.shape[0]) if t.dim() == 0 else t

    def raw(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        The model's raw head at (x, t): ``batch`` with the moved key replaced
        by ``x`` and the time by ``t``, a scalar or one value per row of x.
        """
        inputs = {**batch, self.key: x, self.time_key: self._rows(x, t)}
        return self(inputs)[self.output_key]

    def guidance_score(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor | None:
        """
        The guidance's score term F at (x, t), in the moved key's units, or
        None without guidance. F is the sum of the weighted terms — each
        weight a 1/kT — evaluated on ``batch`` with the path time ``t`` under
        ``time_key``, for guidance that depends on t.
        """
        # TODO: only score guidance for now: every term is a force-like score
        # increment, and velocity and x0 follow from it through the chart.
        # Velocity guidance (e.g. for chart-free flow matching) needs terms
        # that declare their own space, converted per requested field.
        if not self.guidance:
            return None
        inputs = {**batch, self.key: x, self.time_key: self._rows(x, t)}
        field = self.guidance_field(inputs, x * self.position_conversion)
        return self.position_conversion * field.to(x)

    def score(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Score of the marginal p_t at (x, t), plus the guidance's term."""
        raw = self.raw(batch, x, t)
        score = self.parametrization.to_score(self.process, raw, x, t)
        guidance = self.guidance_score(batch, x, t)
        return score if guidance is None else score + guidance

    def x0(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Denoised sample x0 at (x, t), shifted by sigma^2 / a times the guidance."""
        raw = self.raw(batch, x, t)
        x0 = self.parametrization.to_x0(self.process, raw, x, t)
        guidance = self.guidance_score(batch, x, t)
        if guidance is None:
            return x0
        t = self._rows(x, t)
        scale = self.process.sigma(t) ** 2 / self.process.a(t)
        return x0 + expand_t(scale, x) * guidance

    def velocity(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Probability-flow velocity at (x, t), shifted by -1/2 g^2 times the guidance."""
        raw = self.raw(batch, x, t)
        velocity = self.parametrization.to_velocity(self.process, raw, x, t)
        guidance = self.guidance_score(batch, x, t)
        if guidance is None:
            return velocity
        return velocity - 0.5 * expand_t(self.sde.g2(self._rows(x, t)), x) * guidance


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


class EnsembleCalculator(ForceCalculator):
    """
    Run an ensemble of force fields on spk batches, reporting how much it
    disagrees.

    The outputs are the members' mean for every collected property, so any
    driver runs on them exactly as on a single model's. On top of that
    ``outputs[uncertainty_key]`` holds how far the members disagreed, one
    value per structure — which, after a relaxation, tells *which*
    structures went somewhere the models were not trained. Several
    uncertainty functions give a dictionary keyed by class name.

    A physical :class:`ForceCalculator`: it takes batches in Angstrom and
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

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        members = nn.ModuleList(
            [
                member if isinstance(member, nn.Module) else self._load_member(member)
                for member in self._resolve(models)
            ]
        )
        super().__init__(
            NNEnsemble(members, list(properties)),
            kind="physical",
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
