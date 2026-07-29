"""
Reverse-time processes, derived generically from a parametrization and a model.

There is exactly one class here and no hierarchy: reverse processes are never
implemented per schedule. Construct one from the ``(process, parametrization)``
pair and a model; the churn knob selects between the reverse-time SDE
(churn = 1) and the probability-flow ODE (churn = 0).
"""

from typing import Callable

import torch

from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process, expand_t

__all__ = ["ReverseProcess"]


class ReverseProcess:
    """
    Time reversal of a forward process, as a one-parameter family

        dx = [ v(x, t) - 1/2 churn g^2 score(x, t) ] dt + sqrt(churn) g dw,

    for churn in [0, 1]. Every member shares the forward marginals: churn = 1
    is the Anderson (1982) reverse-time SDE, churn = 0 the probability-flow
    ODE, and values in between trade discretization error against stochastic
    churn. It maps onto the usual eta knob by churn = eta^2, since
    v - 1/2 eta^2 g^2 s equals the Anderson drift f x - 1/2 (1 + eta^2) g^2 s.

    Writing the family around the velocity rather than the score is what lets
    churn = 0 use a velocity-predicting model *directly*: the route from a
    velocity back to a score divides by g^2 and blows up as t -> 0, and on the
    ODE path that route is never taken. Flow matching therefore costs nothing
    it shouldn't.

    Exposes the drift/diffusion interface every integrator consumes. Integration
    runs backwards in time, so integrators pass dt < 0.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        model: Callable,
        churn: float = 1.0,
        cond=None,
    ):
        """
        Args:
            process: forward process being reversed; supplies the schedule
                and the endpoint scale
            parametrization: contract between the model output and the fields
            model: callable (x, t, cond) -> raw output in ``parametrization``
            churn: stochasticity in [0, 1]; 1 = reverse SDE, 0 = probability-flow
                ODE. Equals eta^2 of the Anderson family.
            cond: conditioning passed through to the model on every call
        """
        parametrization.validate(process)
        self.process = process
        self.parametrization = parametrization
        self.model = model
        self.churn = churn
        self.cond = cond

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion of the process being reversed, endpoint scale
        included. Integrators that need g^2 read it here.
        """
        return self.process.g2(t)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse drift, shaped like x. Costs one model evaluation."""
        raw = self.model(x, t, self.cond)
        velocity = self.parametrization.to_velocity(self.process, raw, x, t)
        if self.churn == 0.0:
            return velocity
        # Same raw output, second conversion — never a second model call.
        score = self.parametrization.to_score(self.process, raw, x, t)
        return velocity - 0.5 * self.churn * expand_t(self.g2(t), x) * score

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion sqrt(churn) g(t), shaped like t; zero on the ODE."""
        if self.churn == 0.0:
            return torch.zeros_like(t)
        return torch.sqrt(self.churn * self.g2(t))

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Score of the marginal p_t, with ``cond`` bound.

        Integrators normally need only :meth:`drift` and :meth:`diffusion`;
        this exists for the ones that discretize the reverse process in terms
        of the raw score, such as
        :class:`~schnetpack.generative.integrators.ancestral.AncestralDDPM`.
        """
        return self.parametrization.to_score(
            self.process, self.model(x, t, self.cond), x, t
        )
