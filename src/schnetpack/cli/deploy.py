"""spk deploy: compile a trained potential for the LAMMPS pair style
(``configs/deploy.yaml``)."""

import hydra
import torch
from omegaconf import DictConfig

from schnetpack.model_format import load_model


@hydra.main(config_path="../configs", config_name="deploy", version_base="1.2")
def main(config: DictConfig):
    """
    Compile the potential ``model_dir`` (a model folder or a spk train run folder) with
    AOTInductor into the package ``out`` for the LAMMPS pair style in
    ``interfaces/lammps`` (see
    :func:`~schnetpack.interfaces.lammps_interface.deploy_lammps`).

    The package runs only with the torch version and on the device it was compiled
    for, so run this on the machine LAMMPS runs on.
    """
    from schnetpack.interfaces.lammps_interface import deploy_lammps

    model = load_model(config.model_dir, device=config.device)
    deploy_lammps(
        model, config.out, device=config.device, dtype=getattr(torch, config.dtype)
    )
    print(f"stored deployed model at {config.out}.")
