import numpy as np
import jax.numpy as jnp
from ase.io import iread
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm
from ase.db import connect
import jraph

def pad_forces(F, n_max, pad_value=0):
    """
    Padding of the atomic forces. Takes input arrays with shape (B,n,3).

    Args:
        F (Array): Array of atomic forces, shape: (B,n,3)
        n_max (int): Target length.
        pad_value (float): Value used for padding, defaults to 0.

    Returns: New array with padded forces, shape: (B,n_max,3)

    """
    n = F.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(F, ((0, 0), (0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, 0), (0, pad_value)))


def pad_atomic_types(z, n_max, pad_value=0):
    n = z.shape[-1]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(z, ((0, 0), (0, pad_length)), mode='constant', constant_values=((0, 0), (0, pad_value)))


def pad_coordinates(R, n_max, pad_value=0):
    n = R.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(R, ((0, 0), (0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, 0), (0, pad_value)))

@dataclass
class AseDataLoader:
    """ASE data loader. Writes data to speciefied ASE Database from an input file.

    Attributes:
        input_file:
        db_path:
        load_stress:
        load_energy_and_forces:
        neighbors_format: dense or sparse
    """
    input_file: str
    db_path: str = None
    load_stress: bool = False
    load_energy_and_forces: bool = True
    neighbors_format: str = 'dense'

    def _load_all_atoms(self) -> Dict:
        def extract_positions(x: Atoms):
            return x.get_positions()

        def extract_numbers(x: Atoms):
            return x.get_atomic_numbers()

        def extract_energy(x: Atoms):
            return x.get_potential_energy()

        def extract_forces(x: Atoms):
            return x.get_forces()

        def extract_stress(x: Atoms):
            return x.get_stress(voigt=False)

        def extract_pbc(x: Atoms):
            return x.get_pbc()

        def extract_unit_cell(x: Atoms):
            return np.array(x.get_cell(complete=False))

        pos = []
        nums = []

        energies = []
        forces = []
        stress = []

        cell = []
        pbc = []

        n_max = max(set(map(lambda x: len(x.get_atomic_numbers()), iread(self.input_file))))

        print(f"Read data from {self.input_file} ...")
        for a in tqdm(iread(self.input_file)):
            pos += [pad_coordinates(extract_positions(a)[None], n_max=n_max).squeeze(axis=0)]
            nums += [pad_atomic_types(extract_numbers(a)[None], n_max=n_max).squeeze(axis=0)]
            cell += [extract_unit_cell(a)]
            pbc += [extract_pbc(a)]

            if self.load_energy_and_forces:
                energies += [extract_energy(a)]
                forces += [pad_forces(extract_forces(a)[None], n_max=n_max).squeeze(axis=0)]

            if self.load_stress:
                stress += [extract_stress(a)]

        loaded_data = {'R': np.stack(pos, axis=0),
                       'z': np.stack(nums, axis=0),
                       'pbc': np.stack(pbc, axis=0),
                       'unit_cell': np.stack(cell, axis=0)
                       }
        if self.load_stress:
            loaded_data.update({'stress': np.stack(stress, axis=0)})
        if self.load_energy_and_forces:
            loaded_data.update({'E': np.stack(energies, axis=0).reshape(-1, 1),
                                'F': np.stack(forces, axis=0)})

        node_mask = np.where(loaded_data['z'] > 0, True, False)
        loaded_data.update({'node_mask': node_mask})

        print("... done!")

        if self.db_path is not None:
            write_to_ase_db(loaded_data, self.db_path)

        return loaded_data


    def load_all(self):
        return self._load_all_atoms()


def write_to_ase_db(loaded_data, db_path):
    db = connect(db_path)
    # Iterate over loaded_data
    for i in range(len(loaded_data['R'])):
        atoms = Atoms(positions=loaded_data['R'][i], numbers=loaded_data['z'][i], cell=loaded_data['unit_cell'][i], pbc=loaded_data['pbc'][i])
        if 'E' in loaded_data:
            atoms.info['energy'] = loaded_data['E'][i]
        if 'F' in loaded_data:
            atoms.arrays['forces'] = loaded_data['F'][i]
        if 'stress' in loaded_data:
            atoms.info['stress'] = loaded_data['stress'][i]
        db.write(atoms)