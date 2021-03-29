import logging
import os
import io
import re
import shutil
import tarfile
import tempfile
from ase import Atoms
from typing import List, Dict, Tuple, Iterable
from urllib import request as request
from torch.utils.data import random_split

import numpy as np
import pytorch_lightning as pl
from ase.io.extxyz import read_xyz
from tqdm import tqdm

from schnetpack.data import *


class AtomsDataModuleError(Exception):
    pass


class AtomsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datapath: str,
        num_train: int,
        num_val: int,
        num_test: int = -1,
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        transform_fn: Optional[torch.nn.Module] = None,
        train_transform_fn: Optional[torch.nn.Module] = None,
        val_transform_fn: Optional[torch.nn.Module] = None,
        test_transform_fn: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            train_transforms=train_transform_fn or transform_fn,
            val_transforms=val_transform_fn or transform_fn,
            test_transforms=test_transform_fn or transform_fn,
        )
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.datapath, self.format = resolve_format(datapath, format)
        self.load_properties = load_properties

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset(self.datapath, self.format)
        if self.num_test < 0:
            self.num_test = len(self.dataset) - self.num_train - self.num_val

        # TODO: handle IterDatasets
        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            self.dataset, [self.num_train, self.num_val, self.num_test]
        )

    @property
    def train_dataset(self) -> BaseAtomsData:
        if not self.has_prepared_data:
            self.prepare_data()

        if not self.has_setup_fit:
            self.setup(stage="fit")
        return self._train_dataset

    @property
    def val_dataset(self) -> BaseAtomsData:
        if not self.has_prepared_data:
            self.prepare_data()

        if not self.has_setup_fit:
            self.setup(stage="fit")
        return self._val_dataset

    @property
    def test_dataset(self) -> BaseAtomsData:
        if not self.has_prepared_data:
            self.prepare_data()

        if not self.has_setup_fit:
            self.setup(stage="test")
        return self._test_dataset


class QM9(AtomsDataModule):
    """QM9 benchmark database for organic molecules.

    The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
    from including C, O, N, F. This class adds convenient functions to download QM9 from
    figshare and load the data into pytorch.

    References:
        .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"

    def __init__(
        self,
        datapath: str,
        num_train: int,
        num_val: int,
        num_test: int = -1,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        remove_uncharacterized: bool = False,
        transform_fn: Optional[torch.nn.Module] = None,
        train_transform_fn: Optional[torch.nn.Module] = None,
        val_transform_fn: Optional[torch.nn.Module] = None,
        test_transform_fn: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            datapath: path to database (or target directory for download).
            format:
            load_properties: reduced set of properties to be loaded
            remove_uncharacterized: do not include uncharacterized molecules.
            transform_fn:
            train_transform_fn:
            val_transform_fn:
            test_transform_fn:
        """
        super().__init__(
            datapath=datapath,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            format=format,
            load_properties=load_properties,
            transform_fn=transform_fn,
            train_transform_fn=train_transform_fn,
            val_transform_fn=val_transform_fn,
            test_transform_fn=test_transform_fn,
        )

        self.remove_uncharacterized = remove_uncharacterized

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                QM9.A: "GHz",
                QM9.B: "GHz",
                QM9.C: "GHz",
                QM9.mu: "D",
                QM9.alpha: "a0^3",
                QM9.homo: "Ha",
                QM9.lumo: "Ha",
                QM9.gap: "Ha",
                QM9.r2: "a0^2",
                QM9.zpve: "Ha",
                QM9.U0: "Ha",
                QM9.U: "Ha",
                QM9.H: "Ha",
                QM9.G: "Ha",
                QM9.Cv: "cal/mol/K",
            }
            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )

            tmpdir = tempfile.mkdtemp("qm9")
            atomrefs = self._download_atomrefs(tmpdir)
            dataset.update_metadata(atomrefs=atomrefs)

            if self.remove_uncharacterized:
                uncharacterized = self._download_uncharacterized(tmpdir)
            else:
                uncharacterized = None
            self._download_data(tmpdir, dataset, uncharacterized=uncharacterized)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)
            if self.remove_uncharacterized and len(dataset) == 133885:
                raise AtomsDataModuleError(
                    "The dataset at the chosen location contains the uncharacterized 3054 molecules. "
                    + "Choose a different location to reload the data or set `remove_uncharacterized=False`!"
                )
            elif not self.remove_uncharacterized and len(dataset) < 133885:
                raise AtomsDataModuleError(
                    "The dataset at the chosen location does NOT contain the uncharacterized 3054 molecules. "
                    + "Choose a different location to reload the data or set `remove_uncharacterized=True`!"
                )

    def _download_uncharacterized(self, tmpdir):
        logging.info("Downloading list of uncharacterized molecules...")
        at_url = "https://ndownloader.figshare.com/files/3195404"
        tmp_path = os.path.join(tmpdir, "uncharacterized.txt")
        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        uncharacterized = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_atomrefs(self, tmpdir):
        logging.info("Downloading GDB-9 atom references...")
        at_url = "https://ndownloader.figshare.com/files/3195395"
        tmp_path = os.path.join(tmpdir, "atomrefs.txt")
        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        props = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]
        atref = {p: np.zeros((100,)) for p in props}
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                for i, p in enumerate(props):
                    atref[p][z] = float(l.split()[i + 1])
        atref = {k: v.tolist() for k, v in atref.items()}
        return atref

    def _download_data(
        self, tmpdir, dataset: BaseAtomsData, uncharacterized: List[int]
    ):
        logging.info("Downloading GDB-9 data...")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        url = "https://ndownloader.figshare.com/files/3195389"

        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
        )

        property_list = []

        irange = np.arange(len(ordered_files), dtype=np.int)
        if uncharacterized is not None:
            irange = np.setdiff1d(irange, np.array(uncharacterized, dtype=np.int) - 1)

        for i in tqdm(irange):
            xyzfile = os.path.join(raw_path, ordered_files[i])
            properties = {}

            tmp = io.StringIO()
            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(dataset.available_properties, l):
                    properties[pn] = np.array([float(p)])
                for line in lines:
                    tmp.write(line.replace("*^", "e"))

            tmp.seek(0)
            ats: Atoms = list(read_xyz(tmp, 0))[0]
            properties[Structure.Z] = ats.numbers
            properties[Structure.R] = ats.positions
            properties[Structure.cell] = ats.cell
            properties[Structure.pbc] = ats.pbc
            property_list.append(properties)

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")

    def train_dataloader(self):
        return AtomsLoader(self.train_dataset, batch_size=100, num_workers=8)
