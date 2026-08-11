import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional
from urllib import request as request

import numpy as np
from ase import Atoms

import schnetpack.properties as structure
from schnetpack.data.atoms import AtomsDataError, DownloadableASEAtomsData
from schnetpack.transform.base import Transform

__all__ = ["MD17"]


class GDMLDataset(DownloadableASEAtomsData):
    """
    Base class for GDML type data (e.g. MD17 or MD22). Requires a dictionary translating between molecule and filenames
    and an URL under which the molecular datasets can be found.
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
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datasets_dict: dictionary mapping molecule names to dataset names
            download_url: URL where individual molecule datasets can me found
            datapath: path to dataset
            molecule: name of the molecule
            tmpdir: name of temporary directory used for parsing
            atomrefs: properties of free atoms
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """
        self.datasets_dict = datasets_dict
        self.download_url = download_url
        self._native_atomrefs = atomrefs or {}
        self.tmpdir = tmpdir

        if molecule not in self.datasets_dict:
            raise AtomsDataError(f"Molecule {molecule} is not supported!")

        self.molecule = molecule

        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
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

    def _check_db(self) -> None:
        super()._check_db()
        md = self.metadata

        if "molecule" not in md:
            raise AtomsDataError(
                "Not a valid GDML dataset. Metadata must contain `molecule`."
            )

        if md["molecule"] != self.molecule:
            raise AtomsDataError(
                f"The dataset at the given location contains `{md['molecule']}` "
                f"instead of `{self.molecule}`."
            )

    def download(self) -> None:
        tmpdir = tempfile.mkdtemp(self.tmpdir)
        md = self.metadata
        md["atomrefs"] = self._native_atomrefs
        md["molecule"] = self.molecule
        self._set_metadata(md)

        self._download_data(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_data(self, tmpdir) -> None:
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
        self.add_systems(property_list=property_list)
        logging.info("Done.")


class MD17(GDMLDataset):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    The raw data is provided by [#md17_1]_.

    References:

    .. [#md17_1] http://quantum-machine.org/gdml/#datasets
    """

    def __init__(
        self,
        datapath: str,
        molecule: str,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
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
            train_transforms: overrides transform_fn for training.
            val_transforms: overrides transform_fn for validation.
            test_transforms: overrides transform_fn for testing.
            subset_idx: indices of the subset to load.
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...).
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
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            atomrefs=atomrefs,
            **kwargs,
        )
