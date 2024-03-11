import os
from typing import Tuple, Union

import lightning as L
import numpy as np
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from schnetpack.data import ASEAtomsData, AtomsDataModule


def data_preparation(z_list, pos_list, wc_list1, z_wannier: int = 8):
    """
    A method to prepare data for feeding to SchNet that stacks atomic positions and wannier centers across
    different configurations

    Arguments:
        z_list: np.ndarray = atomic_numbers of all atom indices in a given configuration
        pos_list: np.ndarray = position of each atoms across all configurations (configurations,number of atoms,3)
        z_wannier: atomic number of the atom around which wannier centers are to be created
    """

    atoms_list = []
    property_list = []
    for z_mol, positions, wanniers in zip(z_list, pos_list, wc_list1):
        ats = Atoms(positions=positions, numbers=z_mol)

        properties = {
            "wan": wanniers,
            "wc_selector": np.array([1 if z == z_wannier else 0 for z in z_mol]),
        }
        property_list.append(properties)
        atoms_list.append(ats)

    return atoms_list, property_list


class Process_xyz_remsing:
    """
    Arguments:
        file_str: str = path of structure file (eg.position.xyz), POSCAR, CONTCAR
        file_inp: str = path of the input control file with extension inp (for CP2K : "eg. water.inp")
        num_wan: int = number of wannier centers per formula unit. eg. for H20, there are 4 WC around
                        oxygen atom (most electronegative), default = 4
        pbe: bool = True/False, True indicates consideration of periodic boundary condition : default: True
        format_in: str = input file type of first argument (file_str): default = 'xyz'
    """

    def __init__(self, file_str, file_inp, num_wan=4, pbc=True, format_in="xyz"):
        self.file_str = file_str
        self.file_inp = file_inp
        self.num_wan = num_wan
        self.pbc = pbc
        self.format_in = format_in
        # print(lattice_vectors)

    def get_line(self, phrase="ABC"):
        """
        searches for a line containing the phrase in the file and returns first line containing it.
        arguments:
        phrase: str = search phrase on file (file_inp): default='ABC'
        """
        with open(self.file_inp) as f:
            for i, line in enumerate(f):
                if phrase in line:
                    return line

    def get_lattice_vectors(self):
        cell = np.multiply(
            np.eye(3), np.array(self.get_line().split()[-3:], dtype=float).reshape(1, 3)
        )
        return cell

    def get_neigh(
        self,
        coords: np.ndarray,
        r_cut: float,
        pbc: Union[bool, Tuple[bool, bool, bool]] = True,
        cell: np.ndarray = None,
        self_interaction: bool = False,
        periodic_self_interaction: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create neighbor list for all points in a point cloud.

        Args:
            coords: (N, 3) array of positions, where N is the number of points.
            r_cut: cutoff distance for neighbor finding.
            pbc: Whether to use periodic boundary conditions. If a list of bools, then
                each entry corresponds to a supercell vector. If a single bool, then the
                same value is used for all supercell vectors.
            cell: (3, 3) array of supercell vectors. cell[i] is the i-th supercell vector.
                Ignored if `pbc == False` or pbc == None`.
            self_interaction: Whether to include self-interaction, i.e. an atom being the
                neighbor of itself in the neighbor list. Should be False for most
                applications.
            periodic_self_interaction: Whether to include interactions of an atom with its
                periodic images. Should be True for most applications.

        Returns:
            edge_index: (2, num_edges) array of edge indices. The first row contains the
                i atoms (center), and the second row contains the j atoms (neighbor).
            shift_vec: (num_edges, 3) array of shift vectors. The number of cell boundaries
                crossed by the bond between atom i and j. The distance vector between atom
                j and atom i is given by `coords[j] - coords[i] + shift_vec.dot(cell)`.
            num_neigh: (N,) array of the number of neighbors for each atom.
        """
        if isinstance(pbc, bool):
            pbc = [pbc] * 3

        if not np.any(pbc):
            self_interaction = False
            periodic_self_interaction = False

        if cell is None:
            if not np.any(pbc):
                cell = np.eye(3)  # dummy cell to use
            else:
                raise RuntimeError("`cell` vectors not provided")

        (
            first_idx,
            second_idx,
            abs_distance,
            distance_vector,
            shift_vec,
        ) = primitive_neighbor_list(
            "ijdDS",
            pbc=pbc,
            cell=cell,
            positions=coords,
            cutoff=r_cut,
            self_interaction=periodic_self_interaction,
        )

        # remove self interactions
        if periodic_self_interaction and (not self_interaction):
            bad_edge = first_idx == second_idx
            bad_edge &= np.all(shift_vec == 0, axis=1)
            keep_edge = ~bad_edge
            if not np.any(keep_edge):
                raise RuntimeError(
                    "After removing self interactions, no edges remain in this system."
                )
            first_idx = first_idx[keep_edge]
            second_idx = second_idx[keep_edge]
            abs_distance = abs_distance[keep_edge]
            distance_vector = distance_vector[keep_edge]
            shift_vec = shift_vec[keep_edge]

        # number of neighbors for each atom
        num_neigh = np.bincount(first_idx)

        # Some atoms with large index may not have neighbors due to the use of bincount.
        # As a concrete example, suppose we have 5 atoms and first_idx is [0,1,1,3,3,3,3],
        # then bincount will be [1, 2, 0, 4], which means atoms 0,1,2,3 have 1,2,0,4
        # neighbors respectively. Although atom 2 is handled by bincount, atom 4 is not.
        # The below part is to make this work.
        if len(num_neigh) != len(coords):
            extra = np.zeros(len(coords) - len(num_neigh), dtype=int)
            num_neigh = np.concatenate((num_neigh, extra))

        edge_index = np.vstack((first_idx, second_idx))

        return edge_index, num_neigh, abs_distance, distance_vector, shift_vec

    def get_structure(self):
        import ase.io as asi

        cell = self.get_lattice_vectors()
        struct_obj = asi.read(filename=self.file_str, format=self.format_in)
        struct_obj.set_pbc(self.pbc)
        struct_obj.set_cell(cell)
        return struct_obj, cell

    def atom_number_positions(self):
        struct_obj, cell = self.get_structure()
        z_mol = struct_obj.numbers[np.array(np.where(struct_obj.numbers != 0))][0]
        pos_mol = struct_obj.get_positions()[
            np.array(np.where(struct_obj.numbers != 0))
        ][0]
        return z_mol, pos_mol

    def wannier_centers(self):
        import numpy as np

        struct_obj, cell = self.get_structure()

        # The following blocks are mode relevant to water system only. This part needs modification for generalization
        oxy_positions = struct_obj.get_positions()[np.where(struct_obj.numbers == 8)]
        wan_positions = struct_obj.get_positions()[np.where(struct_obj.numbers == 0)]
        coords1 = np.concatenate((oxy_positions, wan_positions), axis=0)
        (
            edge_index,
            num_neigh,
            abs_distance,
            distance_vector,
            shift_vec,
        ) = self.get_neigh(
            coords=coords1,
            r_cut=0.74,
            cell=cell,
            pbc=True,
            self_interaction=False,
            periodic_self_interaction=True,
        )
        wc_neigh=num_neigh[: len(oxy_positions)]
        lst_wan = []
        sum1 = 0
        i: int
        for i, entries in enumerate(num_neigh[: len(oxy_positions)]):
            # Uncomment the following line to define absolute position of wannier center
            # lst1 = (np.sum(distance_vector[sum1 : sum1 + entries], axis=0) / entries).reshape(1, 3) + oxy_positions[i : i + 1,]
            # The following line will define position of wannier center relative to oxygen atom.
            lst1 = (
                np.sum(distance_vector[sum1 : sum1 + entries], axis=0) / entries
            ).reshape(1, 3)
            lst_wan.append(lst1[0])
            sum1 += entries

        lst_wan = np.array(lst_wan)

        return lst_wan, oxy_positions, wc_neigh


    def write_xyz(self, file_out="outfile_ret.xyz", format_out="xyz"):
        import ase.io as asi
        from ase import Atoms

        lst_wan, oxy_positions, wc_neigh = self.wannier_centers()
        # print(z_mol.shape)
        # print(len(pos_oxygen),len(lst_wan))
        new_str = Atoms(
            "O" + str(len(oxy_positions)) + "X" + str(len(lst_wan)),
            np.concatenate((oxy_positions, lst_wan), axis=0),
            pbc=True,
            cell=self.get_lattice_vectors(),
        )
        asi.write(file_out, images=new_str, format=format_out)


def read_data(path):
    """

    Args:
        path: Path to the directory containing the data

    Returns:
    """

    z_list = []
    pos_list = []
    wc_list = []
    na_list = []
    neigh_mismath_list=[]

    lst_dir = [x for x in os.listdir(path) if "." not in x]
    for dir in lst_dir:
        file_path1 = path + str(dir)
        if "W64-bulk-HOMO_centers_s1-1_0.xyz" in os.listdir(file_path1):
            file_str = file_path1 + "/W64-bulk-HOMO_centers_s1-1_0.xyz"
            file_inp = file_path1 + "/water.inp"
            pxr = Process_xyz_remsing(file_str=file_str, file_inp=file_inp)
            # struct_obj, cell = pxr.get_structure()
            lst_wan, oxy_positions, wc_neigh = pxr.wannier_centers()
            if list(np.unique(wc_neigh))==[4]:
                z_mol, pos_mol = pxr.atom_number_positions()

                z_list.append(z_mol)
                pos_list.append(pos_mol)
                wc_list.append(lst_wan)
            else:
                neigh_mismath_list.append(dir)
        else:
            na_list.append(dir)
    print("Number of na:", len(na_list))
    print("Number of neighbor mismatch for given cutoff:", len(neigh_mismath_list))

    z_list = np.asarray(z_list)
    pos_list = np.asarray(pos_list)
    wc_list = np.asarray(wc_list)

    print("z_list.shape", z_list.shape)
    print("pos_list.shape", pos_list.shape)
    print("wc_list.shape", wc_list.shape)

    return z_list, pos_list, wc_list


if __name__ == "__main__":
    path = "/project/wen/sadhik22/model_training/wannier_centers/dataset/train_test_configs_orig/D0/"
    z_list, pos_list, wc_list = read_data(path)

    atoms_list, property_list = data_preparation(z_list, pos_list, wc_list)

    # create database
    processed = "/project/wen/sadhik22/model_training/wannier_centers/schnet_processed/"
    db_path = processed + "wannier_dataset.db"
    split_path = processed + "split.npz"

    if os.path.exists(db_path):
        os.remove(db_path)
    dataset = ASEAtomsData.create(
        db_path,
        distance_unit="Ang",
        property_unit_dict={
            "wan": "Ang",
            "wc_selector": None,
        },
    )
    dataset.add_systems(property_list, atoms_list)
    print("Dataset size:", len(dataset))
    print("Dataset properties:", dataset.available_properties)

    # Create train/val/test split that we can reuse later

    # seed to make deterministic split
    L.seed_everything(35)

    num_data = len(dataset)
    train_size = int(0.8 * num_data)
    val_size = int(0.1 * num_data)
    test_size = num_data - train_size - val_size

    dm = AtomsDataModule(
        db_path,
        batch_size=1,
        num_train=train_size,
        num_val=val_size,
        num_test=test_size,
        split_file=split_path,
    )
    dm.setup()

    print("First 10 train idx:", dm.train_idx[:10])
    print("First 10 val idx:", dm.val_idx[:10])
    print("First 10 test idx:", dm.test_idx[:10])
