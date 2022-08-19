import torch

from schnetpack.nn.radial import GaussianRBF


def test_smear_gaussian_one_distance():
    # case of one distance
    dist = torch.tensor([[[1.0]]])

    smear = GaussianRBF(n_rbf=6, cutoff=5.0, trainable=False)
    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []


def test_smear_gaussian_one_distance_trainable():
    dist = torch.tensor([[[1.0]]])
    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))

    smear = GaussianRBF(n_rbf=6, cutoff=5.0, trainable=True)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    params = list(smear.parameters())
    assert len(params) == 2
    assert len(params[0]) == 6
    assert len(params[1]) == 6


def test_smear_gaussian():
    dist = torch.tensor([[[0.0, 1.0, 1.5], [0.5, 1.5, 3.0]]])
    # smear using 4 Gaussian functions with 1. spacing
    smear = GaussianRBF(start=1.0, cutoff=4.0, n_rbf=4)
    # absolute value of centered distances
    expt = torch.tensor(
        [
            [
                [[1, 2, 3, 4], [0, 1, 2, 3], [0.5, 0.5, 1.5, 2.5]],
                [[0.5, 1.5, 2.5, 3.5], [0.5, 0.5, 1.5, 2.5], [2, 1, 0, 1]],
            ]
        ]
    )
    expt = torch.exp(-0.5 * expt**2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []


def test_smear_gaussian_trainable():
    dist = torch.tensor([[[0.0, 1.0, 1.5, 0.25], [0.5, 1.5, 3.0, 1.0]]])
    # smear using 5 Gaussian functions with 0.75 spacing
    smear = GaussianRBF(start=1.0, cutoff=4.0, n_rbf=5, trainable=True)
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
    expt = torch.exp((-0.5 / 0.75**2) * expt**2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    params = list(smear.parameters())
    assert len(params) == 2
    assert len(params[0]) == 5
    assert len(params[1]) == 5
