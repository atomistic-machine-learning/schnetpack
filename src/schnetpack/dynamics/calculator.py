"""
The two ways a dynamics driver consumes a model.

A calculator turns a model into the field one family of drivers steps on:

- :class:`ForceCalculator` hands the time-free drivers a force at the
  current structures. Its ``kind`` says which: a *physical* force, the
  negative gradient of an energy, in eV/Angstrom; or a *pseudo* force, a
  displacement such as GPFF's 2 (x0 - x), in Angstrom, with no energy
  behind it.
- :class:`GenerativeCalculator` hands out one field at (x, t) — score,
  noise, velocity, x0 or pseudo force, one subclass each — read directly
  from the model's outputs, without conversion.
- :class:`JointCalculator` hands the
  :class:`~schnetpack.dynamics.sample.sampler.Sampler` the model's outputs
  at time t, every field of every generated variable at once.

Both share the plumbing of :class:`Calculator`: the device and dtype, the
neighbor list, the gradient policy, the optional cache of the last outputs,
and the rule that the driver's batch is never written to — the neighbor list
and the model get a shallow copy, so the keys they add never reach it. Both
also take :mod:`~schnetpack.dynamics.guidance` terms, which they add to the
field they return: to the forces, or to the generative field. A driver sees
only the guided field.

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
from schnetpack.dynamics.guidance import Guidance
from schnetpack.model.ensemble import NNEnsemble
from schnetpack.uncertainty import AbsoluteUncertainty, Uncertainty
from schnetpack.units import convert_units

__all__ = [
    "Calculator",
    "ForceCalculator",
    "GenerativeCalculator",
    "ScoreCalculator",
    "NoiseCalculator",
    "VelocityCalculator",
    "X0Calculator",
    "PseudoForceCalculator",
    "JointCalculator",
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
    Run a time-indexed generative model on spk batches and read one output.

    :meth:`field` evaluates the model at (x, t) — ``batch`` with the moved
    key replaced by ``x`` and the time by ``t`` — and returns
    ``outputs[output_key]`` plus the guidance. There are no conversions: the
    model must return the field the calculator stands for. A model that
    returns a raw head under another parametrization is wrapped in
    :class:`~schnetpack.generative.parametrizations.ParametrizedModel` first.

    Guidance — restraints such as
    :class:`~schnetpack.dynamics.guidance.HarmonicRestraint`, or terms
    kT ∇ log p — is added to the field as it is: field + sum_i w_i F_i, so
    the same terms tilt whichever field the calculator returns. The terms see
    the batch with the path time under ``time_key``, so a term may depend on
    t. The guidance is evaluated at the noisy iterate x_t, not at a clean
    estimate.

    The subclasses name the field: :class:`ScoreCalculator`,
    :class:`NoiseCalculator`, :class:`VelocityCalculator`,
    :class:`X0Calculator`, :class:`PseudoForceCalculator`.
    """

    default_output_key: str = ""
    """Model output holding the field, unless overridden at construction."""

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        output_key: str | None = None,
        key: str = properties.R,
        time_key: str = properties.t,
        position_unit: str | float = "Ang",
        **kwargs,
    ):
        """
        Args:
            model: batch -> outputs, or the path to a model saved on disk
            output_key: model output holding the field (default: the
                subclass's :attr:`default_output_key`)
            key: batch key the field is taken with respect to
            time_key: batch key the path time is written to, one value per
                row of ``key``
            position_unit: length unit of ``key``, the training data's. Used
                only to evaluate the guidance in Angstrom.

        Remaining keyword arguments are those of :class:`Calculator`.
        """
        super().__init__(model, **kwargs)
        self.output_key = output_key or self.default_output_key
        self.key = key
        self.time_key = time_key
        self.position_conversion = convert_units(position_unit, "Angstrom")

    @staticmethod
    def _rows(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``t`` as one value per row of ``x``."""
        return t.expand(x.shape[0]) if t.dim() == 0 else t

    def guidance_term(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor | None:
        """
        The weighted sum of the guidance terms at (x, t), in the moved key's
        units, or None without guidance.
        """
        if not self.guidance:
            return None
        inputs = {**batch, self.key: x, self.time_key: self._rows(x, t)}
        field = self.guidance_field(inputs, x * self.position_conversion)
        return self.position_conversion * field.to(x)

    def field(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        The model's ``output_key`` at (x, t) plus the guidance, with ``t`` a
        scalar or one value per row of x.

        Raises:
            KeyError: if the model does not return ``output_key``.
        """
        inputs = {**batch, self.key: x, self.time_key: self._rows(x, t)}
        outputs = self(inputs)
        if self.output_key not in outputs:
            raise KeyError(
                f"{type(self).__name__} expects the model to return "
                f"{self.output_key!r}; got {sorted(outputs)}"
            )
        field = outputs[self.output_key]
        guidance = self.guidance_term(batch, x, t)
        return field if guidance is None else field + guidance


class ScoreCalculator(GenerativeCalculator):
    """The score of the marginal p_t at (x, t)."""

    default_output_key = "score"


class NoiseCalculator(GenerativeCalculator):
    """The expected unit noise E[eps | x_t = x] at (x, t)."""

    default_output_key = "noise"


class VelocityCalculator(GenerativeCalculator):
    """The velocity E[d/dt x_t | x_t = x] at (x, t)."""

    default_output_key = "velocity"


class X0Calculator(GenerativeCalculator):
    """The denoised sample E[x0 | x_t = x] at (x, t)."""

    default_output_key = "x0"


class PseudoForceCalculator(GenerativeCalculator):
    """The pseudo force at (x, t), a displacement toward the clean structure."""

    default_output_key = "pseudo_force"


class JointCalculator(Calculator):
    """
    One model call at time t, read by every update of a
    :class:`~schnetpack.dynamics.sample.sampler.Sampler`.

    Returns the model's outputs as they are: the fields the updates read
    (``score``, ``x0_Z``, ...) are written by the model, one
    :class:`~schnetpack.generative.parametrizations.ParametrizedModel` per
    variable, stacked around one network. The time goes in twice, as
    :class:`~schnetpack.generative.transforms.Diffuse` writes it in
    training: per atom under ``time_key`` and per structure under
    ``structure_time_key``.

    Guidance is added to the outputs named in ``guided_fields``, as
    field + sum_i w_i F_i, evaluated on the positions at the current
    structures and time; see :class:`GenerativeCalculator`.
    """

    def __init__(
        self,
        model: str | Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        time_key: str = properties.t,
        structure_time_key: str = "t_structure",
        guided_fields: Sequence[str] = (),
        position_unit: str | float = "Ang",
        **kwargs,
    ):
        """
        Args:
            model: batch -> outputs with every field the updates read, or
                the path to a model saved on disk
            time_key: batch key for the time per atom
            structure_time_key: batch key for the time per structure
            guided_fields: outputs the guidance is added to, e.g.
                ``("score",)``; needed when there is guidance
            position_unit: length unit of the positions, the training
                data's. Used only to evaluate the guidance in Angstrom.

        Remaining keyword arguments are those of :class:`Calculator`.

        Raises:
            ValueError: if there is guidance but no field to add it to.
        """
        super().__init__(model, **kwargs)
        self.time_key = time_key
        self.structure_time_key = structure_time_key
        self.guided_fields = tuple(guided_fields)
        if self.guidance and not self.guided_fields:
            raise ValueError(
                "JointCalculator adds guidance to the outputs named in "
                "guided_fields, e.g. guided_fields=('score',); got none"
            )
        self.position_conversion = convert_units(position_unit, "Angstrom")

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

    def fields(self, batch: Mapping[str, Any], t: torch.Tensor) -> dict[str, Any]:
        """
        The model's outputs at the structures in ``batch`` and time t, with
        the guidance added to the guided fields.

        Args:
            batch: structures with ``idx_m`` and ``n_atoms``
            t: the time, 0-dim

        Raises:
            KeyError: if the model does not return a guided field.
        """
        inputs = self.with_time(batch, t)
        outputs = dict(self(inputs))
        if not self.guidance:
            return outputs
        positions = inputs[properties.R]
        term = self.guidance_field(inputs, positions * self.position_conversion)
        term = self.position_conversion * term.to(positions)
        for key in self.guided_fields:
            if key not in outputs:
                raise KeyError(
                    f"guidance is added to {key!r}, which the model does not "
                    f"return; got {sorted(outputs)}"
                )
            outputs[key] = outputs[key] + term
        return outputs


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

        Remaining keyword arguments are those of :class:`Calculator`.
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
