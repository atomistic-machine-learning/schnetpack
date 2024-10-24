import logging
import os
import shutil
import tempfile
import tarfile
import json
from typing import List, Optional, Dict
from urllib import request as request
import itertools
import numpy as np
from ase import Atoms
import tensorflow_datasets as tfds
import torch
import schnetpack.properties as structure
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from schnetpack.data import *

__all__ = ["QCML"]


# create batch indices for executing function in multiprocessing
def make_batch_indices(max_,slizes):

    batches = np.linspace(0,max_,slizes,dtype=int)
    n = 0
    indices = []
    for idx,m in enumerate(batches):
        try:
            if m == 0:
                start = 0
            else:
                start = batches[idx]
        
            if m == max_ -1:
                end = -1
            else:
                end = batches[idx+1]
            
            indices.append([start,end])
        except:
            pass
    return indices

# make schema for decoding with given props dict
def make_schema_for_decode_fn(props):
    return {k: tf.io.FixedLenSequenceFeature([], dtype=v, allow_missing=True) for k, v in props.items()}

# modify decode_fn to accept tfrec_format as an argument
def decode_fn(record_bytes, tfrec_format):
    return tf.io.parse_single_example(record_bytes, make_schema_for_decode_fn(tfrec_format))

# modify read_gcs_data to pass tfrec_format to decode_fn
def read_gcs_data(gcs_paths, tfrec_format, AUTO=tf.data.experimental.AUTOTUNE):
    # can read multiple gcs paths in parallel
    data = tf.data.TFRecordDataset(gcs_paths, num_parallel_reads=AUTO)
    
    # pass tfrec_format to the decode_fn using lambda
    data = data.map(lambda record: decode_fn(record, tfrec_format))
    
    return [batch for batch in tqdm(data)]

def update_sample(sample,properties,shapes):

    sample["positions"] = sample["positions"].reshape(sample["atomic_numbers"].shape[0],3)
    
    for k in list(set(shapes) & set(properties)):
        if len(shapes[k]) == 1:
            sample[k] = sample[k].reshape(1,shapes[k][0])
        else:
            sample[k] = sample[k].reshape(*shapes[k])

    atm = Atoms(numbers = sample["atomic_numbers"],positions=sample["positions"])
    props = {key: sample[key] for key in properties if key in sample}

    return props,atm

def process_gcs_path(gcs_path, tfrec_format,properties,shapes):

    #gcs_path, tfrec_format = args
    results = tfds.as_numpy(read_gcs_data(gcs_path,tfrec_format))
    props_list = []
    atms_list = []
    for sample in results:

        props, atm = update_sample(sample,properties,shapes)
        props_list.append(props)
        atms_list.append(atm)

    return atms_list, props_list


def get_tfrec(gcs_path):

    with tf.io.gfile.GFile(gcs_path, 'r') as f:
        json_data = json.load(f)["featuresDict"]["features"]
        available_properties = {}
        shapes = {}

        string_to_tf_dtype = {
        "bool": tf.int64,
        "uint8":tf.int64,
        "float32": tf.float32,
        "float64": tf.float32,
        "int32": tf.int32,
        "int64": tf.int64,
        "string": tf.string
        }
        
        for k in json_data.keys():
            try:
                available_properties[k] = string_to_tf_dtype[json_data[k]["tensor"]["dtype"]]

                shape = json_data[k]["tensor"]["shape"]["dimensions"]
                shapes[k] = [int(n) for n in shape] #json_data[k]["tensor"]["shape"]["dimensions"]
            except:
                # just skipping entries where no tensor dtype is specified
                pass
            
    return available_properties,shapes


class QCML(AtomsDataModule):
    """
    QCML Dataset.

    References:
        .. [#_1] 

    """

    available_properties = {
    'hirshfeld_volumes': 'Bohr^3',                  # Volume in atomic units (Bohr^3)
    'forces': 'Hartree/Bohr',                       # Force in atomic units (Hartree/Bohr)
    'octupole': 'e * Bohr^3',                       # Octupole moment in atomic units
    'quadrupole': 'e * Bohr^2',                     # Quadrupole moment in atomic units
    'has_equal_a_b_electrons': 1.0,                 # No unit, boolean property
    'charge': 1.0,                                  # No unit, integer
    'atomic_numbers': 1.0,                          # No unit, atomic number is an integer
    'hamiltonian_matrix_a': 1.0,                    # Hamiltonian matrix, No unit, alpha spin 
    'hirshfeld_spins': 1.0,                         # No unit, dimensionless spin
    'multiplicity': 1.0,                            # No unit, integer
    'hirshfeld_dipoles': 'e * Bohr',                # Dipole moment in atomic units
    'positions': 'Bohr',                            # Positions in atomic units (Bohr)
    'dipole': 'e * Bohr',                           # Dipole moment in atomic units
    'hirshfeld_volume_ratios': 1.0,                 # No unit, ratio
    'hirshfeld_quadrupoles': 'e * Bohr^2',          # Quadrupole moment in atomic units
    'energy': 'Hartree',                            # Energy in atomic units (Hartree)
    'configuration_parent_seq': 1.0,                # No unit, integer sequence
    'orbital_occupations_a': 1.0,                   # No unit, orbital occupation
    'num_heavy_atoms': 1.0,                         # No unit, integer count of heavy atoms
    'formation_energy': 'Hartree',                  # Energy substracted by reference atomic energies in atomic units (Hartree)
    'hirshfeld_charges': 'e',                       # Charge in atomic units (e)
    'orbital_occupations_b': 1.0,                   # No unit,  occupation, beta spin
    'overlap_matrix': 1.0,                          # No unit, overlap is dimensionless
    'configuration_seq': 1.0,                       # No unit, integer sequence
    'hamiltonian_matrix_b': 'Hartree',              # Hamiltonian matrix, No unit, beta spin
    }


    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = ["formation_energy","forces"],
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        version: Optional[str] = "0.0.3",
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then
                batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers
                (overrides num_workers).
            num_test_workers: Number of test data loader workers
                (overrides num_workers).
            distance_unit: Unit of the atom positions and cell as a string
                (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for
                faster performance.
            split_id: The id of the predefined rMD17 train/test splits (0-4).
        """


        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            **kwargs,
        )
        
        self.version = version


    def prepare_data(self):
        if not os.path.exists(self.datapath):

            # filter out not requested properties and make property unit dict
            property_unit_dict = {k : v for k, v in self.available_properties.items() if k in self.load_properties}

            # make sure charge and spin are always present
            for k in ["charge","multiplicity"]:
                if k not in property_unit_dict:
                    property_unit_dict[k] = self.available_properties[k]

            # update load properties for later shape extraction
            self.load_properties = list(property_unit_dict.keys())

            tmpdir = tempfile.mkdtemp("qcml")

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Bohr",
                property_unit_dict=property_unit_dict,

            )
            dataset.update_metadata(version=self.version)

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)


    def _download_data(
        self,
        tmpdir,
        dataset: BaseAtomsData,
    ):
        
        # access the gcs paths (gcloud auth application-default login)
        # three versions of the dataset are available
        gcs_pattern = f"gs://qcml-external-datasets/tfds/qcml_sample_fixed_columns/{self.version}/*"
        gcs_paths = tf.io.gfile.glob(gcs_pattern)
        # second path contains features shape and dtype info
        tfrec_format,shapes = get_tfrec(gcs_paths[1])
        # to only include paths with property data
        gcs_paths = gcs_paths[2:]
        # hardcoded batchwise collection of properties
        indices = make_batch_indices(4096,50)
        #indices = make_batch_indices(2,2)

        for bidx in indices:

            start,end = (bidx[0],bidx[1])
            logging.info(f"Downloading and processing {start} : {end}")
            batch_files = gcs_paths[start:end]
            logging.info("Extracting data...")
            atm_list,props_list = process_gcs_path(batch_files,tfrec_format,properties=self.load_properties,shapes=shapes)
            logging.info("Write batch of atoms to db...")
            dataset.add_systems(props_list,atm_list)


        logging.info("Done.")
