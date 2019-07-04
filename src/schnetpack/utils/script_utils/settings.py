from schnetpack.datasets import QM9, ANI1, MD17, MaterialsProject,\
    OrganicMaterialsDatabase


__all__ = ["divide_by_atoms", "pooling_mode"]


divide_by_atoms = {
    QM9.A: False,
    QM9.B: False,
    QM9.C: False,
    QM9.mu: True,
    QM9.alpha: True,
    QM9.homo: False,
    QM9.lumo: False,
    QM9.gap: False,
    QM9.r2: False,
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
    QM9.A: "sum",
    QM9.B: "sum",
    QM9.C: "sum",
    QM9.mu: "sum",
    QM9.alpha: "sum",
    QM9.homo: "mean",
    QM9.lumo: "mean",
    QM9.gap: "mean",
    QM9.r2: "mean",
    QM9.zpve: "sum",
    QM9.U0: "sum",
    QM9.U: "sum",
    QM9.H: "sum",
    QM9.G: "sum",
    QM9.Cv: "sum",
    ANI1.energy: "sum",
    MD17.energy: "sum",
    MaterialsProject.EformationPerAtom: "mean",
    MaterialsProject.EPerAtom: "mean",
    MaterialsProject.BandGap: "mean",
    MaterialsProject.TotalMagnetization: "sum",
    OrganicMaterialsDatabase.BandGap: "mean",
}
