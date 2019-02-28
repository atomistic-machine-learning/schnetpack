"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os
import warnings
from base64 import b64encode, b64decode

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Dataset

from schnetpack.environment import SimpleEnvironmentProvider, \
    collect_atom_triples
from .definitions import Structure
from .partitioning import train_test_split

logger = logging.getLogger(__name__)


class AtomsDataError(Exception):
    pass


class AtomsData(Dataset):
    ENCODING = 'utf-8'
    available_properties = None

    def __init__(self, dbpath, subset=None, required_properties=None,
                 environment_provider=SimpleEnvironmentProvider(),
                 collect_triples=False, center_positions=True,
                 load_charge=False):
        self.dbpath = dbpath
        self.subset = subset
        self.required_properties = required_properties
        if required_properties is None:
            self.required_properties = self.available_properties
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centered = center_positions
        self.load_charge = load_charge

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        warnings.warn(
            "create_splits is deprecated, " +
            "use schnetpack.data.train_test_split instead",
            DeprecationWarning
        )
        return train_test_split(self, num_train, num_val, split_file)

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = idx if self.subset is None or len(idx) == 0 \
            else np.array(self.subset)[idx]
        return type(self)(self.dbpath, subidx, self.required_properties,
                          self.environment_provider, self.collect_triples,
                          self.centered, self.load_charge)

    def __len__(self):
        if self.subset is None:
            with connect(self.dbpath) as conn:
                return conn.count()
        return len(self.subset)

    def __getitem__(self, idx):
        at, properties = self.get_properties(idx)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(at)

        properties[Structure.neighbors] = torch.LongTensor(
            nbh_idx.astype(np.int))
        properties[Structure.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        properties['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

        if self.collect_triples:
            nbh_idx_j, nbh_idx_k = collect_atom_triples(nbh_idx)
            properties[Structure.neighbor_pairs_j] = torch.LongTensor(
                nbh_idx_j.astype(np.int))
            properties[Structure.neighbor_pairs_k] = torch.LongTensor(
                nbh_idx_k.astype(np.int))

        return properties

    def _subset_index(self, idx):
        # get row
        if self.subset is None:
            idx = int(idx)
        else:
            idx = int(self.subset[idx])
        return idx

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at

    def get_metadata(self, key):
        with connect(self.dbpath) as conn:
            if key in conn.metadata.keys():
                return conn.metadata[key]
        return None

    def set_metadata(self, metadata):
        with connect(self.dbpath) as conn:
            conn.metadata = metadata

    def _add_system(self, conn, atoms, **properties):

        data = {}

        props = properties.keys() \
            if self.available_properties is None \
            else self.available_properties

        for pname in props:
            try:
                prop = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

            try:
                pshape = prop.shape
                ptype = prop.dtype
            except:
                raise AtomsDataError("Required property `" + pname +
                                     "` has to be `numpy.ndarray`.")

            base64_bytes = b64encode(prop.tobytes())
            base64_string = base64_bytes.decode(AtomsData.ENCODING)
            data[pname] = base64_string
            data['_shape_' + pname] = pshape
            data['_dtype_' + pname] = str(ptype)

        conn.write(atoms, data=data)

    def add_system(self, atoms, **properties):
        with connect(self.dbpath) as conn:
            self._add_system(conn, atoms, **properties)

    def add_systems(self, atoms, property_list):
        with connect(self.dbpath) as conn:
            for at, prop in zip(atoms, property_list):
                self._add_system(conn, at, **prop)

    def get_properties(self, idx):
        idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for pname in self.required_properties:
            # new data format
            try:
                shape = row.data['_shape_' + pname]
                dtype = row.data['_dtype_' + pname]
                prop = np.frombuffer(b64decode(row.data[pname]), dtype=dtype)
                prop = prop.reshape(shape)
            except:
                # fallback for properties stored directly
                # in the row
                if pname in row:
                    prop = row[pname]
                else:
                    prop = row.data[pname]

                try:
                    prop.shape
                except AttributeError as e:
                    prop = np.array([prop], dtype=np.float32)

            properties[pname] = torch.FloatTensor(prop)

        if self.load_charge:
            if Structure.charge in row.data.keys():
                shape = row.data['_shape_' + Structure.charge]
                dtype = row.data['_dtype_' + Structure.charge]
                prop = np.frombuffer(b64decode(row.data[Structure.charge]),
                                     dtype=dtype)
                prop = prop.reshape(shape)
                properties[Structure.charge] = torch.FloatTensor(prop)
            else:
                properties[Structure.charge] = torch.FloatTensor(
                    np.array([0.], dtype=np.float32))

        # extract/calculate structure
        properties[Structure.Z] = torch.LongTensor(at.numbers.astype(np.int))
        positions = at.positions.astype(np.float32)
        if self.centered:
            positions -= at.get_center_of_mass()
        properties[Structure.R] = torch.FloatTensor(positions)
        properties[Structure.cell] = torch.FloatTensor(
            at.cell.astype(np.float32))

        return at, properties

    def get_atomref(self, property):
        """
        Returns atomref for property.

        Args:
            property: property in the qm9 dataset

        Returns:
            list: list with atomrefs
        """
        labels = self.get_metadata('atref_labels')
        if labels is None:
            return None

        col = [i for i, l in enumerate(labels) if l == property]
        assert len(col) <= 1

        if len(col) == 1:
            col = col[0]
            atomref = np.array(self.get_metadata('atomrefs'))[:,
                      col:col + 1]
        else:
            atomref = None

        return atomref


class DownloadableAtomsData(AtomsData):

    def __init__(self, dbpath, subset=None, required_properties=None,
                 environment_provider=SimpleEnvironmentProvider(),
                 collect_triples=False, center_positions=True,
                 load_charge=False, download=False):

        super(DownloadableAtomsData, self).__init__(dbpath, subset,
                                                    required_properties,
                                                    environment_provider,
                                                    collect_triples,
                                                    center_positions,
                                                    load_charge)
        if download:
            self.download()

    def download(self):
        """
        Wrapper function for the download method.
        """
        if os.path.exists(self.dbpath):
            logger.info('The dataset has already been downloaded and stored '
                        'at {}'.format(self.dbpath))
        else:
            logger.info('Starting download')
            folder = os.path.dirname(os.path.abspath(self.dbpath))
            if not os.path.exists(folder):
                os.makedirs(folder)
            self._download()

    def _download(self):
        """
        To be implemented in deriving classes.
        """
        raise NotImplementedError
