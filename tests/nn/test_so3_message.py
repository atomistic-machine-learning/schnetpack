import copy

import pytest
import torch

import schnetpack as spk
import schnetpack.properties as properties
from schnetpack.model import CompiledPotential
from schnetpack.nn.ops import so3_message as ops
from schnetpack.nn.ops.so3_message import so3_message, so3_message_pairs

cuda_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or ops.so3_message_triton is None,
    reason="needs CUDA and Triton",
)
CUTOFF = 4.0


def molecules(n_mol, n_atoms, dtype, device, seed=0):
    """Pairs within the cutoff of ``n_mol`` random molecules, sorted by center."""
    g = torch.Generator().manual_seed(seed)
    R = 3.0 * torch.rand(n_mol * n_atoms, 3, generator=g, dtype=dtype)
    Z = torch.randint(1, 10, (n_mol * n_atoms,), generator=g)
    m = torch.arange(n_mol).repeat_interleave(n_atoms)
    same = m[:, None] == m[None, :]
    close = (R[:, None] - R[None]).norm(dim=-1) < CUTOFF
    idx_i, idx_j = torch.nonzero(same & close & ~torch.eye(len(m), dtype=bool)).T
    batch = {
        properties.Z: Z,
        properties.R: R,
        properties.idx_i: idx_i,
        properties.idx_j: idx_j,
        properties.offsets: torch.zeros(len(idx_i), 3, dtype=dtype),
        properties.idx_m: m,
        properties.n_atoms: torch.full((n_mol,), n_atoms),
    }
    return {k: v.to(device) for k, v in batch.items()}


def so3net_pair(lmax, n_atom_basis, dtype, device):
    """The same SO3net potential with the sparse sum and with the kernel."""
    torch.manual_seed(0)

    def build(fused):
        return spk.model.NeuralNetworkPotential(
            representation=spk.model.SO3net(
                n_atom_basis=n_atom_basis,
                n_interactions=2,
                lmax=lmax,
                radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=CUTOFF),
                cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
                fused_kernel=fused,
            ),
            input_modules=[spk.model.PairwiseDistances()],
            output_modules=[
                spk.model.Atomwise(n_in=n_atom_basis, output_key=properties.energy),
                spk.model.Forces(),
            ],
            input_dtype_str=str(dtype).split(".")[1],
        ).to(device, dtype)

    sparse, fused = build(False), build(True)
    fused.load_state_dict(sparse.state_dict())
    return sparse, fused


def force_loss(model, batch):
    """Energy, forces and the gradients of a force loss for the weights."""
    out = model(copy.copy(batch))
    grads = force_loss_grads(out, model)
    return out[properties.energy].detach(), out[properties.forces].detach(), grads


def force_loss_grads(out, model):
    # the bias of the output layer does not change the forces
    return torch.autograd.grad(
        (out[properties.forces] ** 2).sum(),
        model.parameters(),
        materialize_grads=True,
    )


def assert_matches(sparse, fused, batch):
    energy, forces, grads = force_loss(sparse, batch)
    energy_f, forces_f, grads_f = force_loss(fused, batch)
    torch.testing.assert_close(energy_f, energy)
    torch.testing.assert_close(forces_f, forces)
    for a, b in zip(grads_f, grads):
        torch.testing.assert_close(a, b)


@cuda_kernels
@pytest.mark.parametrize(
    "lmax,n_atom_basis",
    [(0, 16), (1, 16), (2, 16), (2, 15), (2, 70), (2, 128), (3, 16), (4, 16)],
)
def test_kernel_matches_sparse_sum(lmax, n_atom_basis):
    sparse, fused = so3net_pair(lmax, n_atom_basis, torch.float64, "cuda")
    assert_matches(sparse, fused, molecules(3, 6, torch.float64, "cuda"))


@cuda_kernels
def test_kernel_sorts_the_pairs():
    sparse, fused = so3net_pair(2, 16, torch.float64, "cuda")
    batch = molecules(3, 6, torch.float64, "cuda")
    order = torch.randperm(len(batch[properties.idx_i]), device="cuda")
    for key in (properties.idx_i, properties.idx_j, properties.offsets):
        batch[key] = batch[key][order]
    assert_matches(sparse, fused, batch)


@cuda_kernels
def test_padding_pairs_are_skipped():
    """Padding as in a padded batch: an extra atom, far pairs on it at the end."""
    sparse, fused = so3net_pair(2, 16, torch.float64, "cuda")
    batch = molecules(2, 6, torch.float64, "cuda")
    plain = force_loss(fused, batch)

    n_pad = 40
    last = batch[properties.R].shape[0]
    idx_pad = torch.full((n_pad,), last, device="cuda")
    pad_offsets = torch.zeros(n_pad, 3, device="cuda", dtype=torch.float64)
    pad_offsets[:, 0] = 10 * CUTOFF
    padded = dict(batch)
    padded[properties.Z] = torch.cat(
        [batch[properties.Z], batch[properties.Z][:1] * 0 + 1]
    )
    padded[properties.R] = torch.cat(
        [batch[properties.R], batch[properties.R][:1] * 0], dim=0
    )
    padded[properties.idx_i] = torch.cat([batch[properties.idx_i], idx_pad])
    padded[properties.idx_j] = torch.cat([batch[properties.idx_j], idx_pad])
    padded[properties.offsets] = torch.cat([batch[properties.offsets], pad_offsets])
    padded[properties.idx_m] = torch.cat(
        [batch[properties.idx_m], batch[properties.idx_m][-1:] + 1]
    )
    padded[properties.n_atoms] = torch.cat(
        [batch[properties.n_atoms], batch[properties.n_atoms][:1] * 0 + 1]
    )
    assert_matches(sparse, fused, padded)

    forces = force_loss(fused, padded)[1]
    torch.testing.assert_close(forces[:last], plain[1])


@cuda_kernels
def test_self_images_are_real_pairs():
    """Periodic images of an atom have i == j and must not be taken for padding."""
    sparse, fused = so3net_pair(2, 16, torch.float64, "cuda")
    batch = molecules(1, 6, torch.float64, "cuda")
    atoms = torch.arange(6, device="cuda")
    image = torch.zeros(6, 3, device="cuda", dtype=torch.float64)
    image[:, 2] = 0.5 * CUTOFF
    idx_i = torch.cat([batch[properties.idx_i], atoms])
    order = torch.argsort(idx_i, stable=True)
    batch[properties.idx_i] = idx_i[order]
    batch[properties.idx_j] = torch.cat([batch[properties.idx_j], atoms])[order]
    batch[properties.offsets] = torch.cat([batch[properties.offsets], image])[order]
    assert_matches(sparse, fused, batch)


@cuda_kernels
def test_far_pairs_inside_the_list():
    """All pairs of a molecule: those past the cutoff sit between the others."""
    sparse, fused = so3net_pair(2, 16, torch.float64, "cuda")
    batch = molecules(1, 8, torch.float64, "cuda")
    i, j = torch.meshgrid(*[torch.arange(8, device="cuda")] * 2, indexing="ij")
    keep = i != j
    batch[properties.R] = batch[properties.R] * 2
    batch[properties.idx_i], batch[properties.idx_j] = i[keep], j[keep]
    batch[properties.offsets] = torch.zeros(56, 3, device="cuda", dtype=torch.float64)
    Rij = batch[properties.R][j[keep]] - batch[properties.R][i[keep]]
    assert (Rij.norm(dim=1) > CUTOFF).any()
    assert_matches(sparse, fused, batch)


@cuda_kernels
def test_empty_pairs():
    x = torch.ones(2, 9, 16, device="cuda")
    idx = torch.zeros(0, dtype=torch.long, device="cuda")
    w, g = x.new_zeros(0, 3, 16), x.new_zeros(0, 3, 9, 9)
    pairs = so3_message_pairs(idx, idx, x.new_zeros(0, 1, 1), 2)
    torch.testing.assert_close(so3_message(x, w, g, pairs), torch.zeros_like(x))


@cuda_kernels
def test_float32_within_rounding():
    sparse, fused = so3net_pair(2, 32, torch.float32, "cuda")
    batch = molecules(4, 8, torch.float32, "cuda")
    energy, forces, grads = force_loss(sparse, batch)
    energy_f, forces_f, grads_f = force_loss(fused, batch)
    assert (forces_f - forces).norm() / forces.norm() < 1e-4
    for a, b in zip(grads_f, grads):
        assert (a - b).norm() <= 1e-3 * b.norm() + 1e-6


@cuda_kernels
def test_second_derivatives():
    """Each input as its own leaf, all pairs of derivatives."""
    torch.manual_seed(0)
    dtype = torch.float64
    n_a, n_p, n_l, n_s, n_f = 4, 7, 2, 4, 3
    idx_i = torch.sort(torch.randint(0, n_a, (n_p,), device="cuda"))[0]
    idx_j = torch.randint(0, n_a, (n_p,), device="cuda")
    cutoff = torch.ones(n_p, device="cuda", dtype=dtype)
    leaf = lambda *s: torch.randn(  # noqa: E731
        *s, device="cuda", dtype=dtype, requires_grad=True
    )
    inputs = (leaf(n_a, n_s, n_f), leaf(n_p, n_l, n_f), leaf(n_p, n_l, n_s, n_s))
    pairs = so3_message_pairs(idx_i, idx_j, cutoff, n_a)
    fn = lambda x, w, g: so3_message(x, w, g, pairs)  # noqa: E731
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


def test_state_dict_and_script():
    """The option adds nothing to the state dict, and TorchScript takes the sparse sum."""
    sparse, fused = so3net_pair(2, 16, torch.float32, "cpu")
    assert sparse.state_dict().keys() == fused.state_dict().keys()
    representation = fused.representation
    batch = molecules(2, 5, torch.float32, "cpu")
    R_ij = batch[properties.R][batch[properties.idx_j]]
    batch[properties.Rij] = R_ij - batch[properties.R][batch[properties.idx_i]]
    scripted = torch.jit.script(representation)
    out = scripted(dict(batch))
    expected = sparse.representation(dict(batch))
    torch.testing.assert_close(
        out["scalar_representation"], expected["scalar_representation"]
    )


def test_cpu_takes_the_sparse_sum():
    sparse, fused = so3net_pair(2, 16, torch.float64, "cpu")
    assert_matches(sparse, fused, molecules(2, 5, torch.float64, "cpu"))


@cuda_kernels
@pytest.mark.slow
def test_kernel_compiled():
    """Forces and force-loss gradients of the compiled model, on several batch shapes."""
    sparse, fused = so3net_pair(2, 16, torch.float64, "cuda")
    compiled = CompiledPotential(fused)
    batches = [
        molecules(n_mol, n_atoms, torch.float64, "cuda", seed=n_mol)
        for n_mol, n_atoms in [(2, 6), (3, 6), (2, 9)]
    ]

    sparse.train(), fused.train()
    for batch in batches:
        grads = force_loss_grads(compiled(dict(batch)), fused)
        for a, b in zip(grads, force_loss_grads(sparse(dict(batch)), sparse)):
            torch.testing.assert_close(a, b)

    sparse.eval(), fused.eval()
    for batch in batches:
        expected, out = sparse(dict(batch)), compiled(dict(batch))
        for key in (properties.energy, properties.forces):
            torch.testing.assert_close(out[key], expected[key])

    # one graph per mode, reused across batch sizes
    assert len(compiled._graphs) == 2
