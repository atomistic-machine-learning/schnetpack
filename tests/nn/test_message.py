import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import schnetpack as spk
import schnetpack.properties as properties
from schnetpack.nn import message_sum, pair_product, scatter_add
from schnetpack.nn.ops import message

cuda_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or message.message_triton is None,
    reason="needs CUDA and Triton",
)


@pytest.fixture
def kernels_on():
    """The op path (kernels are opt-in); restores the switch afterwards."""
    enabled = message.kernels_enabled()
    message.set_kernels_enabled(True)
    yield
    message.set_kernels_enabled(enabled)


def pairs(n_atoms, n_pairs, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx_i = torch.randint(0, n_atoms, (n_pairs,), generator=g).sort().values
    idx_j = torch.randint(0, n_atoms, (n_pairs,), generator=g)
    return idx_i, idx_j


@pytest.mark.parametrize("trailing", [(16,), (3, 8)])
def test_ops_match_plain_torch(trailing, kernels_on):
    idx_i, idx_j = pairs(7, 30)
    x = torch.randn(7, *trailing)
    w = torch.randn(30, *trailing)
    expected = scatter_add(x[idx_j] * w, idx_i, dim_size=7)
    torch.testing.assert_close(message_sum(x, w, idx_i, idx_j, 7), expected)
    torch.testing.assert_close(pair_product(x, x, idx_i, idx_j), x[idx_i] * x[idx_j])


def test_second_derivatives():
    # the ops themselves: their gradients are written with each other
    idx_i, idx_j = pairs(5, 12)
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(12, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda x, w: message._message_sum_op(x, w, idx_i, idx_j, 5), (x, w)
    )
    assert torch.autograd.gradgradcheck(
        lambda x, y: message._pair_product_op(x, y, idx_i, idx_j), (x, y)
    )


def derivatives(fn, *inputs):
    """Output, first derivatives and the gradient of a first-derivative loss."""
    inputs = [t.detach().requires_grad_() for t in inputs]
    out = fn(*inputs)
    g = torch.autograd.grad((out**2).sum(), inputs, create_graph=True)
    gg = torch.autograd.grad(sum((t**2).sum() for t in g), inputs)
    return [out, *g, *gg]


@cuda_kernels
@pytest.mark.parametrize("trailing", [(128,), (3, 40)])
def test_kernels_match_torch(trailing, kernels_on):
    idx_i, idx_j = (t.cuda() for t in pairs(300, 5000))
    kw = dict(dtype=torch.float64, device="cuda")
    x = torch.randn(300, *trailing, **kw)
    y = torch.randn(300, *trailing, **kw)
    w = torch.randn(5000, *trailing, **kw)
    ms = lambda x, w: message_sum(x, w, idx_i, idx_j, 300)  # noqa: E731
    pp = lambda x, y: pair_product(x, y, idx_i, idx_j)  # noqa: E731

    for fn, args in ((ms, (x, w)), (pp, (x, y))):
        kernel = derivatives(fn, *args)
        message.set_kernels_enabled(False)
        try:
            reference = derivatives(fn, *args)
        finally:
            message.set_kernels_enabled(True)
        for a, b in zip(kernel, reference):
            torch.testing.assert_close(a, b)


def schnet_potential():
    cutoff = 5.0
    return spk.model.NeuralNetworkPotential(
        representation=spk.model.SchNet(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[
            spk.model.Atomwise(n_in=16, output_key=properties.energy),
            spk.model.Forces(),
        ],
    )


def molecules(n_atoms=(5, 7), device="cpu"):
    """All pairs within each molecule of a batch, random positions."""
    torch.manual_seed(0)
    idx_i, idx_j, idx_m, start = [], [], [], 0
    for m, n in enumerate(n_atoms):
        i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        keep = i != j
        idx_i.append(i[keep] + start)
        idx_j.append(j[keep] + start)
        idx_m.append(torch.full((n,), m))
        start += n
    idx_i = torch.cat(idx_i)
    batch = {
        properties.Z: torch.randint(1, 10, (start,)),
        properties.R: torch.rand(start, 3) * 3,
        properties.idx_i: idx_i,
        properties.idx_j: torch.cat(idx_j),
        properties.offsets: torch.zeros(len(idx_i), 3),
        properties.idx_m: torch.cat(idx_m),
        properties.n_atoms: torch.tensor(n_atoms),
    }
    return {k: v.to(device) for k, v in batch.items()}


def test_schnet_potential_scripts(kernels_on):
    model = schnet_potential()
    batch = molecules()
    forces = model(dict(batch))[properties.forces]
    scripted = torch.jit.script(model)
    torch.testing.assert_close(scripted(dict(batch))[properties.forces], forces)


def test_traced_forces_keep_the_ops(kernels_on):
    idx_i, idx_j = pairs(6, 20)
    lin = torch.nn.Linear(3, 8, dtype=torch.float64)

    def energy_forces(R, idx_i, idx_j):
        d = (R[idx_j] - R[idx_i]).norm(dim=1, keepdim=True)
        w = torch.nn.functional.silu(lin(d.expand(-1, 3)))
        E = message_sum(torch.ones(6, 8, dtype=R.dtype), w, idx_i, idx_j, 6).sum()
        (dEdR,) = torch.autograd.grad(E, R, create_graph=True)
        return E, -dEdR

    R = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    gm = make_fx(energy_forces, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
        R, idx_i, idx_j
    )
    targets = {str(n.target) for n in gm.graph.nodes}
    assert "schnetpack.message_sum.default" in targets
    assert "schnetpack.pair_product.default" in targets
    torch.testing.assert_close(
        gm(R, idx_i, idx_j)[1], energy_forces(R, idx_i, idx_j)[1]
    )
