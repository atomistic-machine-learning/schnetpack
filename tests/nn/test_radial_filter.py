import copy

import pytest
import torch

import schnetpack as spk
import schnetpack.properties as properties
from schnetpack.nn.ops import radial_filter
from schnetpack.nn.ops.radial_filter import gaussian_band, gaussian_rbf_linear
from schnetpack.nn.radial import gaussian_rbf

cuda_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or radial_filter.radial_filter_triton is None,
    reason="needs CUDA and Triton",
)


def dense(d, rbf, layer):
    return torch.nn.functional.linear(
        gaussian_rbf(d, rbf.offsets, rbf.widths), layer.weight, layer.bias
    )


def distances(n, dtype, device, cutoff=150.0, seed=0):
    """Mostly short distances, a tail past the cutoff, and the first and last centers."""
    g = torch.Generator().manual_seed(seed)
    x = torch.cat(
        [
            15 * torch.rand(n, generator=g, dtype=dtype),
            1.1 * cutoff * torch.rand(n // 10 + 1, generator=g, dtype=dtype),
            torch.tensor([0.0, cutoff, cutoff + 0.1, cutoff + 5.0], dtype=dtype),
        ]
    )
    return x[torch.randperm(len(x), generator=g)].to(device)[:, None]


def layers(n_rbf, n_out, cutoff, dtype, device):
    rbf = spk.nn.GaussianRBF(n_rbf, cutoff).to(device, dtype)
    layer = spk.nn.Dense(n_rbf, n_out).to(device, dtype)
    torch.nn.init.normal_(layer.bias)
    return rbf, layer


def test_band_of_evenly_spaced_gaussians():
    rbf = spk.nn.GaussianRBF(600, 150.0)
    start, spacing, half_band = gaussian_band(rbf.offsets, rbf.widths)
    assert start == 0.0
    assert spacing == pytest.approx(150.0 / 599)
    # width = spacing: terms 6.5 spacings away are exp(-21.1) < 1e-9
    assert half_band == 6
    assert gaussian_band(rbf.offsets, rbf.widths, tol=1e-3) == (start, spacing, 4)
    uneven = torch.tensor([0.0, 1.0, 3.0])
    assert gaussian_band(uneven, torch.ones(3)) is None
    assert gaussian_band(rbf.offsets, rbf.widths * torch.linspace(1, 2, 600)) is None


def test_dense_without_band_and_on_cpu():
    rbf, layer = layers(50, 7, 10.0, torch.float64, "cpu")
    d = distances(40, torch.float64, "cpu", cutoff=10.0)
    expected = dense(d, rbf, layer)
    args = (d, rbf.offsets, rbf.widths, layer.weight, layer.bias)
    assert torch.equal(gaussian_rbf_linear(*args), expected)
    band = gaussian_band(rbf.offsets, rbf.widths)
    assert torch.equal(gaussian_rbf_linear(*args, band=band), expected)


@cuda_kernels
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_banded_matches_dense(dtype):
    # n_out not a multiple of the 64-wide tiles; distances to past the cutoff
    rbf, layer = layers(600, 200, 150.0, torch.float64, "cuda")
    d = distances(3000, torch.float64, "cuda")
    probe = torch.randn(d.shape[0], 1, 200, device="cuda", dtype=torch.float64)
    expected = dense(d, rbf, layer)
    grads = torch.autograd.grad((expected * probe).sum(), [layer.weight, layer.bias])

    rbf_t, layer_t = copy.deepcopy(rbf).to(dtype), copy.deepcopy(layer).to(dtype)
    band = gaussian_band(rbf_t.offsets, rbf_t.widths)
    out = gaussian_rbf_linear(
        d.to(dtype), rbf_t.offsets, rbf_t.widths, layer_t.weight, layer_t.bias, band
    )
    banded_grads = torch.autograd.grad(
        (out * probe.to(dtype)).sum(), [layer_t.weight, layer_t.bias]
    )
    if dtype == torch.float64:
        torch.testing.assert_close(out, expected)
        for a, b in zip(banded_grads, grads):
            torch.testing.assert_close(a, b)
    else:  # float32 rounding, as the dense layer's own
        scale = expected.abs().max()
        assert (out.double() - expected).abs().max() / scale < 1e-5
        for a, b in zip(banded_grads, grads):
            assert (a.double() - b).abs().max() / b.abs().max() < 1e-5


@cuda_kernels
def test_distance_gradient_takes_the_dense_path():
    rbf, layer = layers(100, 32, 20.0, torch.float64, "cuda")
    d = distances(200, torch.float64, "cuda", cutoff=20.0).requires_grad_()
    band = gaussian_band(rbf.offsets, rbf.widths)
    out = gaussian_rbf_linear(
        d, rbf.offsets, rbf.widths, layer.weight, layer.bias, band
    )
    (grad,) = torch.autograd.grad(out.sum(), d)
    expected = dense(d, rbf, layer)
    (expected_grad,) = torch.autograd.grad(expected.sum(), d)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(grad, expected_grad)


def all_pairs_batch(n_mol, n_atoms, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = n_mol * n_atoms
    a, b = torch.meshgrid(torch.arange(n_atoms), torch.arange(n_atoms), indexing="ij")
    keep = a != b
    base = (torch.arange(n_mol) * n_atoms)[:, None]
    idx_i = (base + a[keep][None]).reshape(-1)
    idx_j = (base + b[keep][None]).reshape(-1)
    R = 2.0 * torch.randn(n, 3, generator=g, dtype=dtype)
    batch = {
        properties.Z: torch.randint(1, 10, (n,), generator=g),
        properties.R: R,
        properties.idx_i: idx_i,
        properties.idx_j: idx_j,
        properties.offsets: torch.zeros(len(idx_i), 3, dtype=dtype),
        properties.idx_m: torch.arange(n_mol).repeat_interleave(n_atoms),
        properties.n_atoms: torch.full((n_mol,), n_atoms),
    }
    return {k: v.to(device) for k, v in batch.items()}


def painn_pair():
    """A PaiNN potential with 150 Gaussians on 30 A and a vector head, and the same
    model with the banded filter."""
    torch.manual_seed(0)

    def build(banded):
        return spk.model.NeuralNetworkPotential(
            representation=spk.model.PaiNN(
                n_atom_basis=32,
                n_interactions=2,
                radial_basis=spk.nn.GaussianRBF(n_rbf=150, cutoff=30.0),
                cutoff_fn=spk.nn.CosineCutoff(30.0),
                norm_epsilon=1.0,
                banded_filter=banded,
            ),
            input_modules=[spk.model.PairwiseDistances()],
            output_modules=[spk.model.AtomwiseVector(n_in=32, output_key="v")],
        )

    dense_model = build(False)
    banded_model = build(True)
    banded_model.load_state_dict(dense_model.state_dict())
    return dense_model, banded_model


def test_painn_banded_filter_needs_even_gaussians():
    with pytest.raises(ValueError, match="banded_filter"):
        spk.model.PaiNN(
            n_atom_basis=8,
            n_interactions=1,
            radial_basis=spk.nn.BesselRBF(n_rbf=10, cutoff=5.0),
            cutoff_fn=spk.nn.CosineCutoff(5.0),
            banded_filter=True,
        )


def test_painn_banded_filter_state_dict_and_script():
    dense_model, banded_model = painn_pair()
    assert dense_model.state_dict().keys() == banded_model.state_dict().keys()
    # TorchScript takes the dense filter (the representation: AtomwiseVector's gated
    # block does not script)
    batch = spk.model.PairwiseDistances()(all_pairs_batch(2, 5, torch.float32, "cpu"))
    scripted = torch.jit.script(banded_model.representation)
    for key in ("scalar_representation", "vector_representation"):
        torch.testing.assert_close(
            scripted(dict(batch))[key], dense_model.representation(dict(batch))[key]
        )


@cuda_kernels
def test_painn_banded_filter_matches_dense():
    dense_model, banded_model = painn_pair()
    dense_model.cuda().double()
    banded_model.cuda().double()
    for n_mol in (1, 6):
        batch = all_pairs_batch(n_mol, 9, torch.float64, "cuda", seed=n_mol)
        results = []
        for model in (dense_model, banded_model):
            model.zero_grad()
            v = model(dict(batch))["v"]
            (
                v * torch.linspace(-1, 1, v.numel(), device="cuda").view_as(v)
            ).sum().backward()
            results.append((v.detach(), [p.grad.clone() for p in model.parameters()]))
        torch.testing.assert_close(results[1][0], results[0][0])
        for a, b in zip(results[1][1], results[0][1]):
            torch.testing.assert_close(a, b)


@cuda_kernels
@pytest.mark.slow
def test_painn_banded_filter_compiled():
    dense_model, banded_model = painn_pair()
    dense_model.cuda()
    banded_model.cuda()
    compiled = spk.model.CompiledPotential(banded_model)
    for n_mol in (1, 6):
        batch = all_pairs_batch(n_mol, 9, torch.float32, "cuda", seed=n_mol)
        expected = dense_model(dict(batch))["v"]
        out = compiled(dict(batch))["v"]
        assert (out - expected).abs().max() / expected.abs().max() < 1e-5
