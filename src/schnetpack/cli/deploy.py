"""spkdeploy: compile a trained potential for the LAMMPS pair style."""

import argparse

import torch

from schnetpack.model_format import load_model


def main():
    """
    Compile a trained potential with AOTInductor into a package for the LAMMPS pair
    style in ``interfaces/lammps`` (see
    :func:`~schnetpack.interfaces.lammps_interface.deploy_lammps`).

    The package runs only with the torch version and on the device it was compiled
    for, so run this on the machine LAMMPS runs on.
    """
    from schnetpack.interfaces.lammps_interface import deploy_lammps

    parser = argparse.ArgumentParser(
        description="Compile a SchNetPack potential for LAMMPS (pair_style schnetpack)."
    )
    parser.add_argument("model_path", help="Path to the trained model, e.g. best_model")
    parser.add_argument(
        "deployed_model_path", help="Where the package is written, ending in .pt2"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device LAMMPS runs the model on, e.g. cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Precision of the deployed model (default: float32)",
    )
    args = parser.parse_args()

    model = load_model(args.model_path, device=args.device)
    deploy_lammps(
        model,
        args.deployed_model_path,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    print(f"stored deployed model at {args.deployed_model_path}.")


if __name__ == "__main__":
    main()
