import torch

import schnetpack as spk
import schnetpack.nn as snn


def test_so3_index_tensors_are_buffers():
    """The index tensors move with the module (.to()), so a GPU model holds no CPU
    tensors, and stay out of the state dict."""
    modules = [
        snn.RealSphericalHarmonics(lmax=2),
        snn.SO3ParametricGatedNonlinearity(n_in=8, lmax=2),
        snn.SO3GatedNonlinearity(lmax=2),
    ]
    for m in modules:
        names = dict(m.named_buffers())
        assert "lidx" in names
        assert not any(k.endswith("lidx") or k.endswith("midx") for k in m.state_dict())
    assert "midx" in dict(modules[0].named_buffers())


def test_so3net_tensors_follow_the_model():
    model = spk.model.SO3net(
        n_atom_basis=8,
        n_interactions=1,
        lmax=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=5, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0),
    )
    meta = torch.device("meta")
    model.to(meta)
    for m in model.modules():
        for v in vars(m).values():
            if isinstance(v, torch.Tensor):
                assert v.device == meta, (
                    f"{type(m).__name__} holds a tensor .to() missed"
                )
