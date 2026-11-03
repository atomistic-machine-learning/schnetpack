"""
Time samplers: where along the path training draws its times.

The schedule fixes what sigma(t) is; the time sampler fixes where the model
spends its capacity. :class:`~schnetpack.generative.transforms.Diffuse` takes
the density as a plain ``(n, device) -> (n,)`` hook, so any callable works. The
densities are stated in t; the noise levels they train on follow from the
schedule's sigma(t). Details: ``docs_new/training.md``.
"""

import abc
import math

import torch

from schnetpack.generative.noising import Interpolant

__all__ = ["TimeSampler", "UniformTimes", "NormalTimes", "LogNormalTimes"]


class TimeSampler(abc.ABC):
    """Draws training times."""

    @abc.abstractmethod
    def __call__(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Args:
            n: number of times to draw (one per structure in ``Diffuse``)
            device: device to draw on

        Returns:
            Times of shape (n,), within the process's [t_min, t_max].
        """
        raise NotImplementedError


class UniformTimes(TimeSampler):
    """
    Uniform on [t_min, t_max]: the default, as an object.

    The process's range stops short of t = 0 because the score target
    diverges there; for noise, x0 and pseudo-force heads ``t_min=0.0`` is
    fine.
    """

    def __init__(
        self,
        process: Interpolant,
        t_min: float | None = None,
        t_max: float | None = None,
    ):
        """
        Args:
            process: process whose time range to draw from
            t_min: override the process's own lower end
            t_max: override the process's own upper end
        """
        self.t_min = process.schedule.t_min if t_min is None else t_min
        self.t_max = process.schedule.t_max if t_max is None else t_max

    def __call__(self, n, device=None):
        return self.t_min + (self.t_max - self.t_min) * torch.rand(n, device=device)


def _bounds(process: Interpolant, t_min, t_max) -> tuple[float, float]:
    """The process's time range, or the given ends."""
    t_min = process.schedule.t_min if t_min is None else t_min
    t_max = process.schedule.t_max if t_max is None else t_max
    if t_min >= t_max:
        raise ValueError(f"empty time range [{t_min}, {t_max}]")
    return t_min, t_max


def _truncated_normal(n: int, device, mean: float, std: float, lo: float, hi: float):
    """
    Draws of N(mean, std^2) truncated to [lo, hi], in float64, by the inverse
    CDF: the distribution that redrawing every miss gives, with a fixed number of
    operations and no host sync (so it can run inside a CUDA graph).
    """

    def cdf(x: float) -> float:
        return 0.5 * math.erfc(-(x - mean) / (std * math.sqrt(2.0)))

    c_lo, c_hi = cdf(lo), cdf(hi)
    if c_hi - c_lo < 1e-6:
        raise RuntimeError(
            f"N({mean:g}, {std:g}^2) almost never lands in [{lo:g}, {hi:g}]: "
            "check the mean against the time range"
        )
    u = c_lo + (c_hi - c_lo) * torch.rand(n, device=device, dtype=torch.float64)
    return (mean + std * torch.special.ndtri(u)).clamp(lo, hi)  # rounding at the ends


class NormalTimes(TimeSampler):
    """
    Normal in t, t ~ N(mean, std^2), truncated to [t_min, t_max].

    The noise levels it trains on are the schedule's sigma(t) at these times;
    on :class:`~schnetpack.generative.schedules.VE`, where log sigma is affine
    in t, that is a log-normal density in sigma.
    """

    def __init__(
        self,
        process: Interpolant,
        mean: float,
        std: float,
        t_min: float | None = None,
        t_max: float | None = None,
    ):
        """
        Args:
            process: process whose time range bounds the draws
            mean: mean of t
            std: standard deviation of t
            t_min: override the process's own lower end
            t_max: override the process's own upper end
        """
        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}")
        self.mean, self.std = mean, std
        self.t_min, self.t_max = _bounds(process, t_min, t_max)

    def __call__(self, n, device=None):
        t = _truncated_normal(n, device, self.mean, self.std, self.t_min, self.t_max)
        return t.to(torch.get_default_dtype())


class LogNormalTimes(TimeSampler):
    """
    Log-normal in t, log t ~ N(mean, std^2), truncated to [t_min, t_max].

    On a schedule whose noise level is proportional to t, such as
    :class:`~schnetpack.generative.schedules.VELinear` (EDM's, where sigma = t),
    this is the log-normal noise density of EDM (Karras et al. 2022) and GPFF:
    log sigma ~ N(m, s^2) is ``mean = m - log(sigma_max / t_max)``, ``std = s``.
    GPFF's (m, s) = (-0.7, 1.2) on ``VELinear(sigma_max=30.0)`` is
    ``mean = -0.7 - log(30)``.
    """

    def __init__(
        self,
        process: Interpolant,
        mean: float,
        std: float,
        t_min: float | None = None,
        t_max: float | None = None,
    ):
        """
        Args:
            process: process whose time range bounds the draws
            mean: mean of log t
            std: standard deviation of log t
            t_min: override the process's own lower end (0 is allowed)
            t_max: override the process's own upper end
        """
        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}")
        self.mean, self.std = mean, std
        self.t_min, self.t_max = _bounds(process, t_min, t_max)
        if self.t_max <= 0.0:
            raise ValueError(f"log-normal times need t_max > 0, got {self.t_max}")

    def __call__(self, n, device=None):
        lo = math.log(self.t_min) if self.t_min > 0.0 else -math.inf
        z = _truncated_normal(n, device, self.mean, self.std, lo, math.log(self.t_max))
        return z.exp().clamp(self.t_min, self.t_max).to(torch.get_default_dtype())
