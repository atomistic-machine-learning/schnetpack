"""
Assembly of reverse-time processes from a parametrization and a model.

Reverse processes are never implemented per schedule. There are exactly two,
split by capability rather than by method:

- :class:`~schnetpack.generative.sde.ReverseSDE` — the Anderson churn family,
  for everything that crosses the (f, g) chart: churn > 0 outright, and even
  churn = 0 when the model's output must be converted to a velocity through
  the score. Constructed from an :class:`~schnetpack.generative.sde.SDE`, so
  it cannot exist for a configuration without the Gaussian kernel.
- :class:`ReverseODE` (here) — continuity-equation transport dx = v dt,
  which exists for *any* endpoint law. It accepts only parametrizations
  whose velocity conversion is chart-free (a velocity head), and is why
  vanilla flow matching and shape-prior velocity sampling never touch the
  chart.

:func:`reverse` picks between them from the assembly — the
``(parametrization, churn)`` pair plus any integrator demand — so callers
keep a single churn knob and invalid assemblies fail at construction with
the chart's own diagnosis. Both classes expose the ``drift``/``diffusion``
interface every integrator consumes; integration runs backwards in time, so
integrators pass dt < 0.
"""

from typing import Callable

import torch

from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process
from schnetpack.generative.sde import ReverseSDE

__all__ = ["ReverseODE", "reverse"]


class ReverseODE:
    """
    Deterministic reverse transport along the learned velocity: dx = v dt.

    The continuity equation makes this valid for any endpoint law — Gaussian
    or structured, chart or no chart (see docs_new/flow_matching_sde.md §8.2)
    — which is exactly why it must not depend on the
    :class:`~schnetpack.generative.sde.SDE`. The price of that generality:
    only parametrizations whose velocity conversion never crosses the chart
    qualify (``velocity_needs_chart`` is False — a velocity head). A score,
    noise or x0 head reaches the velocity through f and g^2, and belongs on
    :class:`~schnetpack.generative.sde.ReverseSDE` even at churn = 0.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        model: Callable,
        cond=None,
    ):
        """
        Args:
            process: forward process being reversed; supplies the schedule
            parametrization: contract between the model output and the
                velocity; must have ``velocity_needs_chart = False``
            model: callable (x, t, cond) -> raw output in ``parametrization``
            cond: conditioning passed through to the model on every call
        """
        parametrization.validate(process)
        if parametrization.velocity_needs_chart:
            raise TypeError(
                f"{type(parametrization).__name__} reaches the velocity "
                "through the (f, g) chart (velocity_needs_chart is True), "
                "so its reverse process is a ReverseSDE — use "
                "reverse(process, parametrization, model, churn=0.0) to "
                "assemble the probability-flow ODE through the chart."
            )
        self.process = process
        self.parametrization = parametrization
        self.model = model
        self.cond = cond

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The learned velocity, shaped like x. Costs one model evaluation."""
        raw = self.model(x, t, self.cond)
        return self.parametrization.to_velocity(self.process, raw, x, t)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Zero — this is the ODE."""
        return torch.zeros_like(t)


def reverse(
    process: Process,
    parametrization: Parametrization,
    model: Callable,
    churn: float = 1.0,
    cond=None,
    require_sde: bool = False,
) -> "ReverseODE | ReverseSDE":
    """
    Assemble the reverse process for a ``(process, parametrization)`` pair.

    Returns the chart-free :class:`ReverseODE` exactly when nothing in the
    assembly needs the chart — churn = 0, a chart-free velocity head, and no
    integrator demand — and a
    :class:`~schnetpack.generative.sde.ReverseSDE` otherwise, acquiring the
    chart via :meth:`~schnetpack.generative.processes.Process.sde` so that a
    configuration without the Gaussian kernel fails here, at assembly, with
    the obstruction named.

    Args:
        process: forward process the model was trained on
        parametrization: contract the model was trained under
        model: callable (x, t, cond) -> raw output in ``parametrization``
        churn: stochasticity in [0, 1]; 1 = reverse SDE, 0 = probability-flow
            ODE. Equals eta^2 of the Anderson family.
        cond: conditioning passed through to the model on every call
        require_sde: force the :class:`ReverseSDE` even at churn = 0 — for
            integrators that discretize through the chart's closed forms
            (ancestral steps) rather than drift/diffusion
    """
    if churn == 0.0 and not parametrization.velocity_needs_chart and not require_sde:
        return ReverseODE(process, parametrization, model, cond=cond)
    return ReverseSDE(process.sde(), parametrization, model, churn=churn, cond=cond)
