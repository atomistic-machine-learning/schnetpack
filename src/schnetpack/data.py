"""
This module contains all functionalities required to load atomistic data, generate batches and compute statistics.
It makes use of the ASE database for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis, Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os

import numpy as np
import torch
from ase.db import connect
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .environment import SimpleEnvironmentProvider, collect_atom_triples


class AtomsData(Dataset):
    """
    Data interface for atomistic systems and properties based on an ASE database.

    Args:
        asedb_path (str): path to ASE database
        subset (list): indices of subset (optional)
        properties (list of str): properties to be loaded (optional)
        environment_provider (schnetpack.environment.EnvironmentProvider): method to collect atom neighbors
        collect_triples (bool): If True, collect atom triple indices (default: False).
        center_positions (bool): If True, substract center of mass from atom positions (default: True).
    """

    def __init__(self, asedb_path, subset=None, properties=[], environment_provider=SimpleEnvironmentProvider(),
                 collect_triples=False, center_positions=True):
        self.asedb_path = asedb_path
        self.asedb = connect(asedb_path)
        self.subset = subset
        self.properties = properties
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centered = center_positions

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]
        return AtomsData(self.asedb_path, subidx, self.properties, self.environment_provider, self.collect_triples,
                         self.centered)

    def __len__(self):
        if self.subset is None:
            return self.asedb.count()
        return len(self.subset)

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        # get row
        if self.subset is None:
            row = self.asedb.get(int(idx) + 1)
        else:
            row = self.asedb.get(int(self.subset[idx]) + 1)
        at = row.toatoms()
        return at

    def __getitem__(self, idx):
        # get row
        if self.subset is None:
            row = self.asedb.get(int(idx) + 1)
        else:
            row = self.asedb.get(int(self.subset[idx]) + 1)
        at = self.get_atoms(idx)

        # extract properties
        properties = {}
        for p in self.properties:
            # Capture exception for ISO17 where energies are stored directly in the row
            if p in row:
                prop = row[p]
            else:
                prop = row.data[p]
            try:
                prop.shape
            except AttributeError as e:
                prop = np.array([prop], dtype=np.float32)
            properties[p] = torch.FloatTensor(prop)

        # extract/calculate structure
        properties[Structure.Z] = torch.LongTensor(at.numbers.astype(np.int))

        positions = at.positions.astype(np.float32)
        if self.centered:
            positions -= at.get_center_of_mass()
        properties[Structure.R] = torch.FloatTensor(positions)

        properties[Structure.cell] = torch.FloatTensor(at.cell.astype(np.float32))

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(idx, at)

        properties[Structure.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
        properties[Structure.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
        properties['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

        if self.collect_triples:
            nbh_idx_j, nbh_idx_k = collect_atom_triples(nbh_idx)
            properties[Structure.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            properties[Structure.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

        return properties

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        """
        Splits the dataset into train/validation/test splits, writes split to an npz file and returns subsets.
        Either the sizes of training and validation split or an existing split file with split indices have to be
        supplied. The remaining data will be used in the test dataset.

        Args:
            num_train (int): number of training examples
            num_val (int): number of validation examples
            split_file (str): Path to split file. If file exists, splits will be loaded.
                              Otherwise, a new file will be created where the generated split is stored.

        Returns:
            schnetpack.data.AtomsData: training dataset
            schnetpack.data.AtomsData: validation dataset
            schnetpack.data.AtomsData: test dataset

        """
        if split_file is not None and os.path.exists(split_file):
            S = np.load(split_file)
            train_idx = S['train_idx'].tolist()
            val_idx = S['val_idx'].tolist()
            test_idx = S['test_idx'].tolist()
        else:
            if num_train is None or num_val is None:
                raise ValueError(
                    'You have to supply either split sizes (num_train / num_val) or an npz file with splits.')

            idx = np.random.permutation(len(self))
            train_idx = idx[:num_train].tolist()
            val_idx = idx[num_train:num_train + num_val].tolist()
            test_idx = idx[num_train + num_val:].tolist()

            if split_file is not None:
                np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        train = self.create_subset(train_idx)
        val = self.create_subset(val_idx)
        test = self.create_subset(test_idx)
        return train, val, test


class StatisticsAccumulator:

    def __init__(self, batch=False, atomistic=False):
        """
        Use the incremental Welford algorithm described in [1]_ to accumulate
        the mean and standard deviation over a set of samples.

        Args:
            batch: If set to true, assumes sample is batch and uses leading
                   dimension as batch size
            atomistic: If set to true, average over atom dimension

        References:
        -----------
        .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        """
        # Initialize state variables
        self.count = 0  # Sample count
        self.mean = 0  # Incremental average
        self.M2 = 0  # Sum of squares of differences
        self.batch = batch
        self.atomistic = atomistic

    def add_sample(self, sample_value):
        """
        Add a sample to the accumulator and update running estimators. Differentiates between different types of samples

        Args:
            sample_value (torch.Tensor): data sample
        """

        # Check different cases
        if not self.batch and not self.atomistic:
            self._add_sample(sample_value)
        elif not self.batch and self.atomistic:
            n_atoms = sample_value.size(0)
            for i in range(n_atoms):
                self._add_sample(sample_value[i, :])
        elif self.batch and not self.atomistic:
            n_batch = sample_value.size(0)
            for i in range(n_batch):
                self._add_sample(sample_value[i, :])
        else:
            n_batch = sample_value.shape[0]
            n_atoms = sample_value.shape[1]
            for i in range(n_batch):
                for j in range(n_atoms):
                    self._add_sample(sample_value[i, j, :])

    def _add_sample(self, sample_value):
        # Update count
        self.count += 1
        delta_old = sample_value - self.mean
        # Difference to old mean
        self.mean += delta_old / self.count
        # Update mean estimate
        delta_new = sample_value - self.mean
        # Update sum of differences
        self.M2 += delta_old * delta_new

    def get_statistics(self):
        """
        Compute statistics of all data collected by the accumulator.

        Returns:
            torch.Tensor: Mean of data
            torch.Tensor: Standard deviation of data
        """
        # Compute standard deviation from M2
        mean = self.mean
        stddev = np.sqrt(self.M2 / self.count)
        # TODO: Should no longer be necessary
        # Convert to torch arrays
        # if type(self.mean) == np.ndarray:
        #    mean = torch.FloatTensor(self.mean)
        #    stddev = torch.FloatTensor(stddev)
        # else:
        #    mean = torch.FloatTensor([self.mean])
        #    stddev = torch.FloatTensor([stddev])
        return mean, stddev


def collate_atoms(examples):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int)
        for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(max_size[prop], np.array(val.size(), dtype=np.int))

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(examples[0][p].type()) for p, size in
        max_size.items()
    }
    has_atom_mask = Structure.atom_mask in batch.keys()
    has_neighbor_mask = Structure.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Structure.neighbor_mask] = torch.zeros_like(batch[Structure.neighbors]).float()
    if not has_atom_mask:
        batch[Structure.atom_mask] = torch.zeros_like(batch[Structure.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Structure.neighbor_pairs_j in properties:
        batch[Structure.neighbor_pairs_mask] = torch.zeros_like(batch[Structure.neighbor_pairs_j]).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Structure.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Structure.neighbor_mask][s] = mask
            batch[Structure.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Structure.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Structure.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Structure.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Structure.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Structure.neighbor_pairs_mask][s] = nbh_idx_j >= 0

    # wrap everything in variables
    batch = {k: Variable(v) for k, v in batch.items()}
    return batch


class AtomsLoader(DataLoader):
    r"""
    Convenience for ``torch.data.DataLoader`` which already uses the correct collate_fn for AtomsData and
    provides functionality for calculating mean and stddev.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch (default: collate_atons).
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=collate_atoms, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(AtomsLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                          num_workers, collate_fn, pin_memory, drop_last,
                                          timeout, worker_init_fn)

    def get_statistics(self, property_name, atomistic=False, atomref=None, split_file=None):
        """
        Compute mean and variance of a property.
        Uses the incremental Welford algorithm implemented in StatisticsAccumulator

        Args:
            property_name (str):  Name of the property for which the mean and standard
                            deviation should be computed
            atomistic (bool): If set to true, averages over atoms
            atomref (np.ndarray): atomref (default: None)
            split_file (str): path to split file. If specified, mean and std will be cached in this file (default: None)

        Returns:
            mean:           Mean value
            stddev:         Standard deviation

        """
        with torch.no_grad():
            calc_stats = True
            if split_file is not None and os.path.exists(split_file):
                split_data = np.load(split_file)

                if 'mean' in split_data and 'stddev' in split_data:
                    mean = torch.from_numpy(split_data['mean'])
                    stddev = torch.from_numpy(split_data['stddev'])
                    calc_stats = False
                    logging.info("cached statistics was loaded...")

            if calc_stats:
                statistics = StatisticsAccumulator(batch=True)
                logging.info("statistics will be calculated...")

                count = 0
                for row in self:
                    self._update_statistic(atomistic, atomref, property_name, row, statistics)
                    count += 1

                mean, stddev = statistics.get_statistics()

                # cache result in split file
                if split_file is not None and os.path.exists(split_file):
                    split_data = np.load(split_file)
                    np.savez(split_file, train_idx=split_data['train_idx'],
                             val_idx=split_data['val_idx'], test_idx=split_data['test_idx'], mean=mean, stddev=stddev)

            return mean, stddev

    def _update_statistic(self, atomistic, atomref, property_name, row, statistics):
        """
        Helper function to update iterative mean / stddev statistics computation
        """
        property_value = row[property_name]
        if atomref is not None:
            z = row['_atomic_numbers']
            p0 = torch.sum(torch.from_numpy(atomref[z]).float(), dim=1)
            property_value -= p0
        if atomistic:
            property_value /= torch.sum(row['_atom_mask'], dim=1, keepdim=True)
        statistics.add_sample(property_value)


class Structure:
    """
    Keys to access structure properties loaded using `schnetpack.data.AtomsData`
    """
    Z = '_atomic_numbers'
    atom_mask = '_atom_mask'
    R = '_positions'
    cell = '_cell'
    neighbors = '_neighbors'
    neighbor_mask = '_neighbor_mask'
    cell_offset = '_cell_offset'
    neighbor_pairs_j = '_neighbor_pairs_j'
    neighbor_pairs_k = '_neighbor_pairs_k'
    neighbor_pairs_mask = '_neighbor_pairs_mask'
