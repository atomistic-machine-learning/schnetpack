import torch

from schnetpack.nn.acsf import GaussianSmearing


def test_smear_gaussian_one_distance():
    # case of one distance
    dist = torch.tensor([[[1.0]]])
    # trainable = False
    smear = GaussianSmearing(n_gaussians=6, centered=False, trainable=False)
    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []
    # trainable = True
    smear = GaussianSmearing(n_gaussians=6, centered=False, trainable=True)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    params = list(smear.parameters())
    assert len(params) == 2
    assert len(params[0]) == 6
    assert len(params[1]) == 6
    # centered = True
    smear = GaussianSmearing(n_gaussians=6, centered=True)
    expt = -0.5 / torch.tensor([0.0, 1, 2, 3, 4, 5]) ** 2
    expt = torch.exp(expt * dist[:, :, :, None] ** 2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []


def test_smear_gaussian():
    dist = torch.tensor([[[0.0, 1.0, 1.5], [0.5, 1.5, 3.0]]])
    # smear using 4 Gaussian functions with 1. spacing
    smear = GaussianSmearing(start=1.0, stop=4.0, n_gaussians=4)
    # absolute value of centered distances
    expt = torch.tensor(
        [
            [
                [[1, 2, 3, 4], [0, 1, 2, 3], [0.5, 0.5, 1.5, 2.5]],
                [[0.5, 1.5, 2.5, 3.5], [0.5, 0.5, 1.5, 2.5], [2, 1, 0, 1]],
            ]
        ]
    )
    expt = torch.exp(-0.5 * expt ** 2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []
    # centered = True
    smear = GaussianSmearing(start=1.0, stop=4.0, n_gaussians=4, centered=True)
    expt = torch.exp(
        (-0.5 / torch.tensor([1, 2, 3, 4.0]) ** 2) * dist[:, :, :, None] ** 2
    )
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []


def test_smear_gaussian_trainable():
    dist = torch.tensor([[[0.0, 1.0, 1.5, 0.25], [0.5, 1.5, 3.0, 1.0]]])
    # smear using 5 Gaussian functions with 0.75 spacing
    smear = GaussianSmearing(start=1.0, stop=4.0, n_gaussians=5, trainable=True)
    # absolute value of centered distances
    expt = torch.tensor(
        [
            [
                [
                    [1, 1.75, 2.5, 3.25, 4.0],
                    [0, 0.75, 1.5, 2.25, 3.0],
                    [0.5, 0.25, 1.0, 1.75, 2.5],
                    [0.75, 1.5, 2.25, 3.0, 3.75],
                ],
                [
                    [0.5, 1.25, 2.0, 2.75, 3.5],
                    [0.5, 0.25, 1.0, 1.75, 2.5],
                    [2.0, 1.25, 0.5, 0.25, 1.0],
                    [0, 0.75, 1.5, 2.25, 3.0],
                ],
            ]
        ]
    )
    expt = torch.exp((-0.5 / 0.75 ** 2) * expt ** 2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    params = list(smear.parameters())
    assert len(params) == 2
    assert len(params[0]) == 5
    assert len(params[1]) == 5
    # centered = True
    smear = GaussianSmearing(
        start=1.0, stop=4.0, n_gaussians=5, trainable=True, centered=True
    )
    expt = -0.5 / torch.tensor([1, 1.75, 2.5, 3.25, 4]) ** 2
    expt = torch.exp(expt * dist[:, :, :, None] ** 2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    params = list(smear.parameters())
    assert len(params) == 2
    assert len(params[0]) == 5
    assert len(params[1]) == 5
