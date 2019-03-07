import numpy as np
import pytest
import torch
from pytest import approx

from schnetpack.representation import GDML


@pytest.fixture
def dummy_gdml_model():
    return {
        'sig': 52,
        'c': 1.234,
        'std': 9.9,
        'perms': np.array([[0, 1, 2], [0, 2, 1]]),
        'tril_perms_lin': np.array([[0, 3, 1, 5, 2, 4]]),
        'R_desc': 1 + np.arange(12, dtype=float).reshape(3, 4),
        'R_d_desc_alpha': np.arange(12, dtype=float).reshape(4, 3),
    }


def test_gdml_dummy(dummy_gdml_model):
    Rs = torch.arange(18, dtype=torch.double).reshape(2, 3, 3)
    Es, Fs = GDML(dummy_gdml_model)(Rs)
    assert Es.shape == (2,)
    assert Es[0].item() == approx(-4.10982937)
    assert Fs.shape == (2, 3, 3)
    assert Fs[0, 1, 1].item() == approx(-0.00060433, abs=1e-7)
    assert Fs[1, 2, 1].item() == approx(0.00542786, abs=1e-7)
