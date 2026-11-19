"""A linear layer over Gaussian radial basis functions, evaluated on each pair's band.

``gaussian_rbf_linear(d, offsets, widths, weight, bias, band)`` computes
``F.linear(gaussian_rbf(d, offsets, widths), weight, bias)``: the filter network of
PaiNN (and the first layer of SchNet's) on a Gaussian expansion of the pair distances.

Gaussians that are evenly spaced and as wide as their spacing overlap only their
neighbors: a pair's expansion is nonzero on a band of about a dozen centers around its
distance, whatever the number of Gaussians. With ``band`` (from :func:`gaussian_band`)
each pair sums only the rows of ``weight^T`` within ``half_band`` of its nearest
center::

    out[p] = bias + sum_{|k - k_p| <= half_band} exp(coeff_k (d_p - mu_k)^2) weight[:, k]

Each term is the dense layer's (``mu_k``, ``coeff_k = -0.5 / width_k^2`` from the
buffers); the dropped terms are below ``tol`` (default 1e-9) times their weight. With
600 Gaussians on 150 A (GPFF's PaiNN) a training step's filter (forward and weight
gradient, 157k pairs x 3072 features) takes 3.3 ms instead of 32 ms on an H100 PCIe.

The banded op runs on CUDA through Triton kernels
(:mod:`schnetpack.nn.ops.radial_filter_triton`) and has gradients for ``weight`` and
``bias`` only. Everything else takes the dense path: CPU tensors, TorchScript, no
Triton, no bias, and distances or Gaussians that need a gradient (forces as the
derivative of the energy, trainable RBFs).
"""

import math

import torch
import torch.nn.functional as F

from schnetpack.nn.radial import gaussian_rbf

__all__ = ["gaussian_rbf_linear", "gaussian_band"]


def gaussian_band(
    offsets: torch.Tensor, widths: torch.Tensor, tol: float = 1e-9
) -> tuple[float, float, int] | None:
    """
    The band of Gaussians each pair needs: ``(start, spacing, half_band)``.

    Args:
        offsets: centers of the Gaussians, increasing
        widths: their widths
        tol: largest dropped term, relative to its weight

    Returns:
        ``offsets[0]``, the spacing and the number of rows on each side of a pair's
        nearest center, or None if the centers are not evenly spaced or the widths
        differ (then every Gaussian may matter).
    """
    if offsets.numel() < 2:
        return None
    steps = (offsets[1:] - offsets[:-1]).detach().double().cpu()
    w = widths.detach().double().cpu()
    spacing = float(steps.mean())
    if spacing <= 0.0 or not (
        torch.allclose(steps, torch.full_like(steps, spacing), rtol=1e-4)
        and torch.allclose(w, torch.full_like(w, float(w[0])))
    ):
        return None
    # a term (half_band + 1/2) spacings from the pair is exp(-x^2 / 2) with
    # x = (half_band + 1/2) spacing / width; keep every term above tol
    half_band = math.ceil(math.sqrt(-2.0 * math.log(tol)) * float(w[0]) / spacing - 0.5)
    return float(offsets[0]), spacing, max(half_band, 0)


def _dense(
    d: torch.Tensor,
    offsets: torch.Tensor,
    widths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(gaussian_rbf(d, offsets, widths), weight, bias)


try:
    from schnetpack.nn.ops import radial_filter_triton
except ImportError:  # no Triton: the dense path on CUDA as well
    radial_filter_triton = None

if radial_filter_triton is not None:
    _op = torch.library.triton_op(
        "schnetpack::gaussian_rbf_linear",
        radial_filter_triton.gaussian_rbf_linear_triton,
        mutates_args=(),
    )
    _grad_op = torch.library.triton_op(
        "schnetpack::gaussian_rbf_linear_grad",
        radial_filter_triton.gaussian_rbf_linear_grad_triton,
        mutates_args=(),
    )

    def _setup(ctx, inputs, output):
        x, order, weight, bias, offsets, coeff, start, spacing, half_band = inputs
        ctx.save_for_backward(x, order, offsets, coeff)
        ctx.band = (start, spacing, half_band)

    def _backward(ctx, grad):
        x, order, offsets, coeff = ctx.saved_tensors
        if any(ctx.needs_input_grad[i] for i in (0, 4, 5)):
            raise RuntimeError(
                "gaussian_rbf_linear: the banded op has no gradient for the distances "
                "or the Gaussians; these inputs take the dense path"
            )
        gwt, gbias = _grad_op(x, order, grad, offsets, coeff, *ctx.band)
        return None, None, gwt.t(), gbias, None, None, None, None, None

    _op.register_autograd(_backward, setup_context=_setup)


@torch.jit.unused
def _banded(
    d: torch.Tensor,
    offsets: torch.Tensor,
    widths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    band: tuple[float, float, int],
) -> torch.Tensor:
    if (
        radial_filter_triton is None
        or bias is None
        or d.requires_grad
        or offsets.requires_grad
        or widths.requires_grad
        or not radial_filter_triton.supported(d, weight)
    ):
        return _dense(d, offsets, widths, weight, bias)
    x = d.reshape(-1)
    order = torch.argsort(x.detach())  # a tile of sorted pairs spans a few Gaussians
    coeff = -0.5 / torch.pow(widths, 2)
    out = _op(x, order, weight, bias, offsets, coeff, *band)
    return out.view(d.shape + (weight.shape[0],))


def gaussian_rbf_linear(
    d: torch.Tensor,
    offsets: torch.Tensor,
    widths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    band: tuple[float, float, int] | None = None,
) -> torch.Tensor:
    """
    ``F.linear(gaussian_rbf(d, offsets, widths), weight, bias)``, on each pair's band of
    Gaussians when ``band`` is given and the inputs allow it (module docstring).

    Args:
        d: distances, any shape
        offsets: centers of the Gaussians, shape ``(n_rbf,)``
        widths: their widths, shape ``(n_rbf,)``
        weight: weight of the linear layer, shape ``(n_out, n_rbf)``
        bias: its bias, shape ``(n_out,)``, or None
        band: :func:`gaussian_band` of offsets and widths; None for the dense layer

    Returns:
        shape ``d.shape + (n_out,)``
    """
    if band is None or torch.jit.is_scripting():
        return _dense(d, offsets, widths, weight, bias)
    return _banded(d, offsets, widths, weight, bias, band)
