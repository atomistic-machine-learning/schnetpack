import logging
import os
import shutil
import tempfile
import zipfile
from typing import List, Optional, Dict
from urllib import request as request

from ase.io import read
from ase import Atoms
import numpy as np

import torch
import schnetpack.properties as structure
from matbench_discovery.data import ase_atoms_from_zip
from schnetpack.data import *
import ase

__all__ = ["MPTraj"]


class MPTraj(AtomsDataModule):
    """
    MPTRJ Dataset loader (custom .extxyz inside .zip) using SchNetPack.
    """

    energy = "energy"
    forces = "forces"
    stress = "stress"
    MaterialId = "material_id"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
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
        **kwargs,
    ):
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

        # Dataset specific configuration
        self.datasets_dict = {
            "mptrj": "mptrj",
        }
        self.download_url = "https://figshare.com/files/43302033" # "https://figshare.com/files/49034296"
        self.molecule = "mptrj"
        self.tmpdir = "mptrj_tmp"
        self.atomrefs = {
            self.energy: [0.0]
            * 119  # Replace with real atom reference values if available
        }

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                self.energy: "eV/atom",
                self.forces: "eV/Ang",
                self.stress: "eV/Ang^3",
                self.MaterialId: None,
            }

            tmpdir = tempfile.mkdtemp(self.tmpdir)

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=self.atomrefs,
            )
            dataset.update_metadata(molecule=self.molecule)

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data(self, tmpdir, dataset: BaseAtomsData):
        filename = self.datasets_dict[self.molecule]
        url = self.download_url
        local_path = os.path.join(tmpdir, os.path.basename(filename))
        print(local_path)

        logging.info(f"Downloading {filename} from {url}...")
        request.urlretrieve(url, local_path)

        logging.info("Loading structures from zip file...")
        atoms_list = self._load_atoms_from_zip(local_path)

        property_list = []
        key_value_pairs_list = []
        for atoms in atoms_list:
            # Convert energy to eV/atom by dividing by number of atoms
            total_energy = atoms.get_total_energy()
            num_atoms = len(atoms)
            energy_per_atom = total_energy / num_atoms
            
            properties = {
                self.energy: np.array([energy_per_atom]),
                self.forces: atoms.get_forces(),
                self.stress: atoms.get_stress(),
                structure.Z: atoms.get_atomic_numbers(),
                structure.R: atoms.get_positions(),
                structure.cell: atoms.get_cell(),
                structure.pbc: atoms.get_pbc(),
            }

            property_list.append(properties)
            key_value_pairs_list.append({"material_id": atoms.info.get("material_id")})

        logging.info("Write atoms to db...")
        dataset.add_systems(
            property_list=property_list,
            key_value_list=key_value_pairs_list,
        )
        logging.info("Done.")

    def _load_atoms_from_zip(self, zip_path):
        """
        Load atoms from a zip file containing .extxyz files.
        Handles different encoding issues that might occur with the zip file.
        """
        atoms_list = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # List all files in the zip
                file_list = zip_file.namelist()
                logging.info(f"Found {len(file_list)} files in zip")
                
                for filename in file_list:
                    if filename.endswith('.extxyz') or filename.endswith('.xyz'):
                        try:
                            # Try to read the file content
                            with zip_file.open(filename) as file:
                                # Read as bytes first
                                content = file.read()
                                
                                # Try different encodings
                                encodings = ['utf-8', 'latin-1', 'cp1252']
                                content_str = None
                                
                                for encoding in encodings:
                                    try:
                                        content_str = content.decode(encoding)
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                
                                if content_str is None:
                                    logging.warning(f"Could not decode {filename} with any encoding, skipping")
                                    continue
                                
                                # Parse the content using ASE
                                try:
                                    # Use StringIO to create a file-like object
                                    from io import StringIO
                                    atoms = read(StringIO(content_str), format='extxyz')
                                    atoms_list.append(atoms)
                                except Exception as e:
                                    logging.warning(f"Could not parse {filename}: {e}")
                                    continue
                                    
                        except Exception as e:
                            logging.warning(f"Error reading {filename}: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"Error opening zip file {zip_path}: {e}")
            raise
            
        logging.info(f"Successfully loaded {len(atoms_list)} structures")
        return atoms_list
