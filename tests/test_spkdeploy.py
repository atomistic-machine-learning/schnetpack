import os
import subprocess
import sys

import torch
from ase.io import read

import schnetpack as spk

TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")
SPKDEPLOY = os.path.join(os.path.dirname(__file__), "../src/scripts/spkdeploy")


def test_spkdeploy_exports_a_model_saved_with_an_older_version(tmp_path):
    """md_ethanol.model predates 2.1.0 and only scripts after the conversion that
    schnetpack.utils.load_model applies."""
    model_path = os.path.join(TESTDATA, "md_ethanol.model")
    deployed_path = tmp_path / "deployed.pt"
    subprocess.run(
        [sys.executable, SPKDEPLOY, model_path, str(deployed_path)],
        check=True,
        capture_output=True,
    )

    metadata = {"cutoff": ""}
    deployed = torch.jit.load(str(deployed_path), _extra_files=metadata)
    model = spk.utils.load_model(model_path)
    assert float(metadata["cutoff"]) == model.representation.cutoff.item()

    # the deployed model runs in float32 without the CastTo64 postprocessor
    atoms = read(os.path.join(TESTDATA, "md_ethanol.xyz"))
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=spk.transform.ASENeighborList(cutoff=5.0)
    )
    expected = model(converter(atoms))
    results = deployed(converter(atoms))
    for key in ["energy", "forces"]:
        torch.testing.assert_close(
            results[key].double(), expected[key].double(), rtol=1e-6, atol=1e-4
        )
