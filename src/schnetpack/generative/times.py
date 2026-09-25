"""
Time samplers: where along the path training draws its times.

The schedule fixes what sigma(t) is; the time sampler fixes where the model
spends its capacity. Both :class:`~schnetpack.generative.losses.MatchingLoss`
and :class:`~schnetpack.generative.transforms.Diffuse` take the density as a
plain ``(n, device) -> (n,)`` hook, so any callable works. Details:
``docs_new/training.md``.
"""

import abc
import math

import torch

from schnetpack.generative.processes import Process

__all__ = ["TimeSampler", "UniformTimes", "LogNormalSigmaTimes"]


class TimeSampler(abc.ABC):
    """Draws training times."""

    @abc.abstractmethod
    def __call__(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Args:
            n: number of times to draw (one per sample in ``MatchingLoss``,
                one per structure in ``Diffuse``)
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
        process: Process,
        t_min: float | None = None,
        t_max: float | None = None,
    ):
        """
        Args:
            process: process whose time range to draw from
            t_min: override the process's own lower end
            t_max: override the process's own upper end
        """
        self.t_min = process.t_min if t_min is None else t_min
        self.t_max = process.t_max if t_max is None else t_max

    def __call__(self, n, device=None):
        return self.t_min + (self.t_max - self.t_min) * torch.rand(n, device=device)


class LogNormalSigmaTimes(TimeSampler):
    """
    Log-normal in the noise level, log sigma ~ N(mean, std), mapped to t
    through :meth:`~schnetpack.generative.processes.Process.t_of_sigma`
    (the EDM / GPFF training density).

    Defaults are GPFF's (``mean=-0.7, std=1.2``): median sigma ~0.5 A. The
    right mean for a dataset is set by its length scale. Draws outside
    [sigma_min, sigma_max] are clamped onto the bounds, or rejected and
    redrawn with ``truncate=True`` (GPFF's choice at its upper end).
    """

    def __init__(
        self,
        process: Process,
        mean: float = -0.7,
        std: float = 1.2,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        truncate: bool = False,
    ):
        """
        Args:
            process: process whose sigma(t) defines the map to times; needs
                a prior with a declared ``std``
            mean: mean of log sigma, in the data's length units
            std: standard deviation of log sigma
            sigma_min: lower bound on the draws (default: sigma(t_min))
            sigma_max: upper bound on the draws (default: sigma(t_max))
            truncate: reject and redraw outside [sigma_min, sigma_max]
                rather than clamping onto it
        """
        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}")
        _ = process.std  # raises with the useful message if the prior has no scale
        self.process = process
        self.mean = mean
        self.std = std
        self.truncate = truncate

        ends = torch.tensor([process.t_min, process.t_max])
        s_low, s_high = torch.sort(process.sigma(ends)).values.tolist()
        self.sigma_min = s_low if sigma_min is None else sigma_min
        self.sigma_max = s_high if sigma_max is None else sigma_max
        if self.sigma_min >= self.sigma_max:
            raise ValueError(f"empty sigma range [{self.sigma_min}, {self.sigma_max}]")

    def _draw_sigma(self, n: int, device) -> torch.Tensor:
        return torch.exp(
            self.mean + self.std * torch.randn(n, device=device, dtype=torch.float64)
        )

    def __call__(self, n, device=None):
        sigma = self._draw_sigma(n, device)
        if self.truncate:
            # Resample only the rows that missed, so every draw stays an
            # independent sample of the truncated law. Bounded because each
            # round keeps a fixed fraction; the cap is a guard against a
            # range so far into the tail that acceptance is ~0.
            for _ in range(100):
                bad = (sigma < self.sigma_min) | (sigma > self.sigma_max)
                if not bool(bad.any()):
                    break
                sigma = torch.where(
                    bad, self._draw_sigma(n, device).to(sigma.dtype), sigma
                )
            else:
                raise RuntimeError(
                    f"log-normal sigma (mean={self.mean}, std={self.std}) almost "
                    f"never lands in [{self.sigma_min:g}, {self.sigma_max:g}] — "
                    "check the mean against the data's length scale"
                )
        else:
            sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        return self.process.t_of_sigma(sigma.to(torch.get_default_dtype()))

    def induced_normal(self) -> tuple:
        """
        The (mean, std) of t this density induces on a geometric
        :class:`~schnetpack.generative.processes.VE`, where t is affine in
        log sigma. Raises TypeError for other schedules.
        """
        from schnetpack.generative.processes import VE

        if not isinstance(self.process, VE):
            raise TypeError(
                f"t is affine in log sigma only on a geometric VE schedule, "
                f"not {type(self.process).__name__} — the induced density over "
                "t has no closed form there; draw and measure instead."
            )
        log_range = -math.log(self.process.b_min)
        t_mean = self.process.t_max * (
            1.0 + (self.mean - math.log(self.process.std)) / log_range
        )
        return t_mean, self.process.t_max * self.std / log_range
