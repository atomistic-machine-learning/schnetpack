import logging
import os
import shutil
import tempfile
from typing import List, Optional, Dict
from urllib import request as request
import torch

import numpy as np
from ase import Atoms

import schnetpack.properties as structure
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError

__all__ = ["MD17"]


class GDMLDataset(ASEAtomsData):
    """
    Base class for GDML-type datasets (e.g. MD17 or MD22).
    Requires a dictionary translating between molecule and filenames
    and a URL under which the molecular datasets can be found.
    """

    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        datasets_dict: Dict[str, str],
        download_url: str,
        datapath: str,
        molecule: str,
        tmpdir: str = "gdml_tmp",
        atomrefs: Optional[Dict[str, List[float]]] = None,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datasets_dict: dictionary mapping molecule names to dataset names.
            download_url: URL where individual molecule datasets can me found.
            datapath: path to dataset.
            molecule: name of the molecule.
            tmpdir: name of temporary directory used for parsing.
            atomrefs: properties of free atoms.
            load_properties: subset of properties to load.
            transforms: Transform applied to each system separately before batching.
            subset_idx: indices of the subset to load.
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...).
            **kwargs: additional keyword arguments.
        """
        self.datasets_dict = datasets_dict
        self.download_url = download_url
        self._native_atomrefs = atomrefs
        self.tmpdir = tmpdir

        if molecule not in self.datasets_dict:
            raise AtomsDataError(f"Molecule {molecule} is not supported!")

        self.molecule = molecule

        self.download(
            datapath=datapath,
            distance_unit=distance_unit or "Ang",
        )

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        return {
            GDMLDataset.energy: "kcal/mol",
            GDMLDataset.forces: "kcal/mol/Ang",
        }

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        if os.path.exists(datapath):
            dataset = ASEAtomsData(datapath, load_structure=False)
            md = dataset.metadata

            if "molecule" not in md:
                raise AtomsDataError(
                    "Not a valid GDML dataset. Metadata must contain `molecule`."
                )

            if md["molecule"] != self.molecule:
                raise AtomsDataError(
                    f"The dataset at the given location contains `{md['molecule']}` "
                    f"instead of `{self.molecule}`."
                )
            return

        tmpdir = tempfile.mkdtemp(self.tmpdir)
        try:
            dataset = ASEAtomsData.create(
                datapath=datapath,
                distance_unit=distance_unit,
                property_unit_dict=self._native_property_units(),
                atomrefs=self._native_atomrefs,
            )
            dataset.update_metadata(molecule=self.molecule)
            self._download_data(tmpdir, dataset)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_data(
        self,
        tmpdir,
        dataset: ASEAtomsData,
    ):
        logging.info("Downloading {} data".format(self.molecule))
        rawpath = os.path.join(tmpdir, self.datasets_dict[self.molecule])
        url = self.download_url + self.datasets_dict[self.molecule]

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(rawpath)

        numbers = data["z"]
        property_list = []
        for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
            ats = Atoms(positions=positions, numbers=numbers)
            properties = {
                self.energy: (
                    energies if type(energies) is np.ndarray else np.array([energies])
                ),
                self.forces: forces,
                structure.Z: ats.numbers,
                structure.R: ats.positions,
                structure.cell: ats.cell,
                structure.pbc: ats.pbc,
            }
            property_list.append(properties)

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")


class MD17(GDMLDataset):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    References:
        .. [#md17_1] http://quantum-machine.org/gdml/#datasets
    """

    def __init__(
        self,
        datapath: str,
        molecule: str,
        load_properties: Optional[List[str]] = None,
        transforms=None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset.
            molecule: name of the molecule.
            load_properties: subset of properties to load.
            transforms: Transform applied to each system separately before batching.
            subset_idx: indices of the subset to load.
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...).
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

        datasets_dict = dict(
            aspirin="md17_aspirin.npz",
            azobenzene="azobenzene_dft.npz",
            benzene="md17_benzene2017.npz",
            ethanol="md17_ethanol.npz",
            malonaldehyde="md17_malonaldehyde.npz",
            naphthalene="md17_naphthalene.npz",
            paracetamol="paracetamol_dft.npz",
            salicylic_acid="md17_salicylic.npz",
            toluene="md17_toluene.npz",
            uracil="md17_uracil.npz",
        )

        super().__init__(
            datasets_dict=datasets_dict,
            download_url="http://www.quantum-machine.org/gdml/data/npz/",
            tmpdir="md17",
            molecule=molecule,
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            atomrefs=atomrefs,
            **kwargs,
        )
