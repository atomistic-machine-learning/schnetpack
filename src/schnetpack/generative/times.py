"""
Time samplers — where along the path training spends its samples.

The counterpart of :mod:`~schnetpack.generative.grids`: a grid decides where a
*sampler* puts its steps on the way back, a time sampler decides where
*training* draws its times on the way out. Both distribute attention over the
same axis, and both are swappable without touching anything else, because the
consumers — :class:`~schnetpack.generative.losses.MatchingLoss` and
:class:`~schnetpack.generative.transforms.Diffuse` — take the density as a
plain ``(n, device) -> (n,)`` hook.

Why it is worth choosing rather than defaulting: the schedule fixes what
sigma(t) *is*, but not where the model spends its capacity. Uniform t on a
geometric VE schedule is log-uniform in sigma, spreading samples evenly over
orders of magnitude — including the deep-noise end where the target is nearly
the endpoint itself and there is little to learn. The EDM/GPFF answer is to
state the density in *sigma*, concentrating on the middle band where denoising
is actually hard, which is what :class:`LogNormalSigmaTimes` does.

Anything matching the hook shape works — a bare lambda is fine. These classes
exist to name the two densities that recur, and to keep the sigma-space one
honest about the process it is tied to.
"""

import abc
import math
from typing import Optional

import torch

from schnetpack.generative.processes import Process

__all__ = ["TimeSampler", "UniformTimes", "LogNormalSigmaTimes"]


class TimeSampler(abc.ABC):
    """Draws training times."""

    @abc.abstractmethod
    def __call__(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Args:
            n: number of times to draw. Note the two callers differ:
                ``MatchingLoss`` asks for one per sample, ``Diffuse`` asks
                for a single time per structure and broadcasts it over the
                atoms — so a sampler must not assume a batch.
            device: device to draw on

        Returns:
            Times of shape (n,), within the process's [t_min, t_max].
        """
        raise NotImplementedError


class UniformTimes(TimeSampler):
    """
    Uniform on [t_min, t_max] — the default, as an object.

    Identical to :meth:`~schnetpack.generative.processes.Process.sample_t`,
    which is what the consumers fall back to when given no sampler. Worth
    having explicitly so a config can *name* the default alongside the
    alternatives, and so the range can be widened: the default stops short of
    t = 0 because the score target diverges there, but the noise, denoiser and
    pseudo-force targets are well behaved, and for those ``t_min=0.0`` is
    available.
    """

    def __init__(
        self,
        process: Process,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
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
    Log-normal in the noise level: log sigma ~ N(mean, std), mapped to t.

    The EDM (Karras et al. 2022) and GPFF training density. It is stated in
    sigma because that is where the statement is meaningful — "train mostly
    around half an Angstrom of displacement" is a claim about geometry, and
    stays that claim whatever schedule carries it. The process converts, via
    :meth:`~schnetpack.generative.processes.Process.t_of_sigma`.

    On a geometric :class:`~schnetpack.generative.processes.VE` schedule t is
    affine in log sigma, so the induced density over t is exactly normal::

        t ~ N( t_max (1 + (mean - log sigma_max) / L),  t_max std / L ),
        L = log(sigma_max / sigma_min)

    — a fact worth knowing when reading the numbers, but not one this class
    relies on: it draws in sigma and converts, so a non-geometric schedule
    gets the same *sigma* density rather than the same t density.

    Defaults are GPFF's (``mean=-0.7``, ``std=1.2``): a median sigma of
    ~0.5 A, most of the mass between 0.15 and 1.7 A. EDM's own choice for
    images is ``mean=-1.2, std=1.2`` on data scaled to unit variance. Both are
    statements about *their* data — the right mean for a new dataset is set by
    the scale where its structure lives, not inherited.

    A log-normal has tails at both ends and the schedule does not, so the two
    have to be reconciled. By default draws outside [sigma_min, sigma_max] are
    **clamped** onto the bounds, which piles a little mass exactly there;
    ``truncate=True`` **rejects and redraws** them instead, which is what GPFF
    does at its upper end and leaves the interior density untouched. The
    choice only matters when the density is wide relative to the range — with
    GPFF's defaults on ``VE(0.05, 30)``, ~3% of draws fall below sigma_min.
    """

    def __init__(
        self,
        process: Process,
        mean: float = -0.7,
        std: float = 1.2,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        truncate: bool = False,
    ):
        """
        Args:
            process: process whose sigma(t) defines the map to times; it must
                have a scalar noise scale (a prior with a declared ``std``)
            mean: mean of log sigma, in the data's length units
            std: standard deviation of log sigma
            sigma_min: lower bound on the draws; defaults to the schedule's
                own sigma(t_min)
            sigma_max: upper bound on the draws; defaults to the schedule's
                own sigma(t_max). GPFF's rejection bound.
            truncate: reject and redraw outside [sigma_min, sigma_max] rather
                than clamping onto it
        """
        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}")
        process.std  # raises with the useful message if the prior has no scale
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
        The (mean, std) of t this density induces, where t is affine in log
        sigma — i.e. on a geometric :class:`VE`. Convenience for reading a
        configuration; raises where the shape does not hold.
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
