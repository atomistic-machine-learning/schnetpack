from typing import Optional, Dict, List

from schnetpack.data import AtomsDataFormat
from schnetpack.datasets.md17 import GDMLDataset

__all__ = ["MD22"]


class MD22(GDMLDataset):
    """
    MD22 benchmark data set for extended molecules containing molecular forces.

    References:
        .. [#md22_1] http://quantum-machine.org/gdml/#datasets
    """

    def __init__(
        self,
        datapath: str,
        molecule: str,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        transforms=None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            molecule: name of the molecule
            format: dataset format
            load_properties: subset of properties to load
            transforms: Transform applied to each system separately before batching.
            subset_idx: indices of the subset to load.
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            **kwargs: additional keyword arguments.
        """
        atomrefs = {
            self.energy: [
                0.0,
                -313.5150902000774,
                0.0,
                0.0,
                0.0,
                0.0,
                -23622.587180094913,
                -34219.46811826416,
                -47069.30768969713,
            ]
        }

        datasets_dict = {
            "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
            "DHA": "md22_DHA.npz",
            "stachyose": "md22_stachyose.npz",
            "AT-AT": "md22_AT-AT.npz",
            "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
            "buckyball-catcher": "md22_buckyball-catcher.npz",
            "double-walled_nanotube": "md22_double-walled_nanotube.npz",
        }

        super().__init__(
            datasets_dict=datasets_dict,
            download_url="http://www.quantum-machine.org/gdml/repo/datasets/",
            tmpdir="md22",
            molecule=molecule,
            datapath=datapath,
            format=format,
            load_properties=load_properties,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            atomrefs=atomrefs,
            **kwargs,
        )
