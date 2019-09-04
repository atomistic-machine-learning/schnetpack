import logging
import os
import shutil
import tempfile
from urllib import request as request

import numpy as np
from ase import Atoms

import schnetpack as spk
from schnetpack.data import AtomsDataError
from schnetpack.datasets import DownloadableAtomsData

__all__ = ["MD17"]


class MD17(DownloadableAtomsData):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    Args:
        dbpath (str): path to database
        molecule (str): Name of molecule to load into database. Allowed are:
                            aspirin
                            benzene
                            ethanol
                            malonaldehyde
                            naphthalene
                            salicylic_acid
                            toluene
                            uracil
        subset (list): indices of subset. Set to None for entire dataset
            (default: None)
        download (bool): set true if dataset should be downloaded
            (default: True)
        collect_triples (bool): set true if triples for angular functions
            should be computed (default: False)
        load_only (list, optional): reduced set of properties to be loaded
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).


    See: http://quantum-machine.org/datasets/
    """

    energy = "energy"
    forces = "forces"

    datasets_dict = dict(
        aspirin="aspirin_dft.npz",
        # aspirin_ccsd='aspirin_ccsd.zip',
        azobenzene="azobenzene_dft.npz",
        benzene="benzene_dft.npz",
        ethanol="ethanol_dft.npz",
        # ethanol_ccsdt='ethanol_ccsd_t.zip',
        malonaldehyde="malonaldehyde_dft.npz",
        # malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
        naphthalene="naphthalene_dft.npz",
        paracetamol="paracetamol_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        # toluene_ccsdt='toluene_ccsd_t.zip',
        uracil="uracil_dft.npz",
    )

    existing_datasets = datasets_dict.keys()

    def __init__(
        self,
        dbpath,
        molecule=None,
        subset=None,
        download=True,
        collect_triples=False,
        load_only=None,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
    ):
        if not os.path.exists(dbpath) and molecule is None:
            raise AtomsDataError("Provide a valid dbpath or select desired molecule!")

        if molecule is not None and molecule not in MD17.datasets_dict.keys():
            raise AtomsDataError("Molecule {} is not supported!".format(molecule))

        self.molecule = molecule

        available_properties = [MD17.energy, MD17.forces]

        super(MD17, self).__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            collect_triples=collect_triples,
            download=download,
            available_properties=available_properties,
            environment_provider=environment_provider,
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return MD17(
            dbpath=self.dbpath,
            molecule=self.molecule,
            subset=subidx,
            download=False,
            collect_triples=self.collect_triples,
            load_only=self.load_only,
            environment_provider=self.environment_provider,
        )

    def _download(self):

        logging.info("Downloading {} data".format(self.molecule))
        tmpdir = tempfile.mkdtemp("MD")
        rawpath = os.path.join(tmpdir, self.datasets_dict[self.molecule])
        url = (
            "http://www.quantum-machine.org/gdml/data/npz/"
            + self.datasets_dict[self.molecule]
        )

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(rawpath)

        numbers = data["z"]
        atoms_list = []
        properties_list = []
        for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
            properties_list.append(dict(energy=energies, forces=forces))
            atoms_list.append(Atoms(positions=positions, numbers=numbers))

        self.add_systems(atoms_list, properties_list)
        self.update_metadata(dict(data_source=self.datasets_dict[self.molecule]))

        logging.info("Cleanining up the mess...")
        logging.info("{} molecule done".format(self.molecule))
        shutil.rmtree(tmpdir)
