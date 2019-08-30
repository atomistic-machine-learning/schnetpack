import schnetpack as spk
from schnetpack.datasets import (
    QM9,
    ANI1,
    MD17,
    MaterialsProject,
    OrganicMaterialsDatabase,
)


__all__ = [
    "divide_by_atoms",
    "pooling_mode",
    "get_divide_by_atoms",
    "get_pooling_mode",
    "get_negative_dr",
    "get_derivative",
]


divide_by_atoms = {
    QM9.mu: True,
    QM9.alpha: True,
    QM9.homo: False,
    QM9.lumo: False,
    QM9.gap: False,
    QM9.r2: True,
    QM9.zpve: True,
    QM9.U0: True,
    QM9.U: True,
    QM9.H: True,
    QM9.G: True,
    QM9.Cv: True,
    ANI1.energy: True,
    MD17.energy: True,
    MaterialsProject.EformationPerAtom: False,
    MaterialsProject.EPerAtom: False,
    MaterialsProject.BandGap: False,
    MaterialsProject.TotalMagnetization: True,
    OrganicMaterialsDatabase.BandGap: False,
}

pooling_mode = {
    QM9.mu: "sum",
    QM9.alpha: "sum",
    QM9.homo: "avg",
    QM9.lumo: "avg",
    QM9.gap: "avg",
    QM9.r2: "sum",
    QM9.zpve: "sum",
    QM9.U0: "sum",
    QM9.U: "sum",
    QM9.H: "sum",
    QM9.G: "sum",
    QM9.Cv: "sum",
    ANI1.energy: "sum",
    MD17.energy: "sum",
    MaterialsProject.EformationPerAtom: "avg",
    MaterialsProject.EPerAtom: "avg",
    MaterialsProject.BandGap: "avg",
    MaterialsProject.TotalMagnetization: "sum",
    OrganicMaterialsDatabase.BandGap: "avg",
}


def get_divide_by_atoms(args):
    """
    Get 'divide_by_atoms'-parameter depending on run arguments.
    """
    if args.dataset == "custom":
        return args.aggregation_mode == "sum"
    return divide_by_atoms[args.property]


def get_pooling_mode(args):
    """
    Get 'pooling_mode'-parameter depending on run arguments.
    """
    if args.dataset == "custom":
        return args.aggregation_mode
    return pooling_mode[args.property]


def get_derivative(args):
    if args.dataset == "custom":
        return args.derivative
    elif args.dataset == "md17" and not args.ignore_forces:
        return spk.datasets.MD17.forces
    return None


def get_negative_dr(args):
    if args.dataset == "custom":
        return args.negative_dr
    elif args.dataset == "md17":
        return True
    return False
