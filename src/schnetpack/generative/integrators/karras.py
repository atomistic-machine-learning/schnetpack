"""Karras stochastic sampling (EDM Algorithm 2)."""

from schnetpack.generative.integrators.heun import Heun

__all__ = ["KarrasStochasticHeun"]


class KarrasStochasticHeun(Heun):
    """
    Heun with Karras-style churn: inflate the noise level, then step back down.

    Each step first raises the state from sigma to sigma_hat = sigma (1 + gamma)
    by adding matching noise, then takes a deterministic Heun step from
    sigma_hat to the next grid point. Unlike the generic churn knob, which mixes
    stochasticity into the drift continuously, this alternates exact noise
    injection with exact deterministic transport — the noise corrects drift
    accumulated by the ODE solver rather than diffusing alongside it.

    Deferred, for two reasons that are worth recording:

    - it needs ``gamma = min(S_churn / n_steps, sqrt(2) - 1)``, so it must
      override :meth:`~schnetpack.generative.integrators.base.Integrator.integrate`
      to see the step count rather than just :meth:`step`;
    - inverting sigma_hat back to a path time is only trivial where t *is*
      sigma, which makes it EDM-specific rather than generic.

    It lives here rather than as a Sampler or a TimeGrid on purpose: it rewrites
    what a step computes, which is exactly what an Integrator owns. A grid is a
    static sequence of times and cannot express noise injection; a dedicated
    sampler would fork the single composition path the design exists to protect.

    Until this lands, use ``churn > 0`` for stochastic sampling, or
    :class:`~schnetpack.generative.integrators.heun.Heun` with ``churn = 0`` for
    the deterministic EDM sampler (Algorithm 1), which the generic machinery
    already reproduces exactly.
    """

    def step(self, process, x, t, dt):
        raise NotImplementedError(
            "KarrasStochasticHeun (EDM Alg. 2) is not implemented yet. "
            "Use Heun with churn=0 for the deterministic EDM sampler, or "
            "EulerMaruyama/Heun with churn>0 for generic stochastic sampling."
        )

    def integrate(self, process, x, ts):
        raise NotImplementedError(
            "KarrasStochasticHeun (EDM Alg. 2) is not implemented yet. "
            "Use Heun with churn=0 for the deterministic EDM sampler, or "
            "EulerMaruyama/Heun with churn>0 for generic stochastic sampling."
        )
