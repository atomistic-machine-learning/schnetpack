"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from typing import Final

idx: Final[str] = "_idx"

## structure
Z: Final[str] = "_atomic_numbers"  #: nuclear charge
position: Final[str] = "_positions"  #: atom positions
R: Final[str] = position  #: atom positions

cell: Final[str] = "_cell"  #: unit cell
strain: Final[str] = "strain"
pbc: Final[str] = "_pbc"  #: periodic boundary conditions

seg_m: Final[str] = "_seg_m"  #: start indices of systems
idx_m: Final[str] = "_idx_m"  #: indices of systems
idx_i: Final[str] = "_idx_i"  #: indices of center atoms
idx_j: Final[str] = "_idx_j"  #: indices of neighboring atoms
idx_i_lr: Final[str] = "_idx_i_lr"  #: indices of center atoms for long-range
idx_j_lr: Final[str] = "_idx_j_lr"  #: indices of neighboring atoms for long-range

lidx_i: Final[str] = "_idx_i_local"  #: local indices of center atoms (within system)
lidx_j: Final[
    str
] = "_idx_j_local"  #: local indices of neighboring atoms (within system)
Rij: Final[str] = "_Rij"  #: vectors pointing from center atoms to neighboring atoms
Rij_lr: Final[
    str
] = "_Rij_lr"  #: vectors pointing from center atoms to neighboring atoms for long range
n_atoms: Final[str] = "_n_atoms"  #: number of atoms
offsets: Final[str] = "_offsets"  #: cell offset vectors
offsets_lr: Final[str] = "_offsets_lr"  #: cell offset vectors for long range

R_strained: Final[str] = (
    position + "_strained"
)  #: atom positions with strain-dependence
cell_strained: Final[str] = cell + "_strained"  #: atom positions with strain-dependence

n_nbh: Final[str] = "_n_nbh"  #: number of neighbors

#: indices of center atom triples
idx_i_triples: Final[str] = "_idx_i_triples"

#: indices of first neighboring atom triples
idx_j_triples: Final[str] = "_idx_j_triples"

#: indices of second neighboring atom triples
idx_k_triples: Final[str] = "_idx_k_triples"

## chemical properties
energy: Final[str] = "energy"
forces: Final[str] = "forces"
stress: Final[str] = "stress"
masses: Final[str] = "masses"
dipole_moment: Final[str] = "dipole_moment"
polarizability: Final[str] = "polarizability"
hessian: Final[str] = "hessian"
dipole_derivatives: Final[str] = "dipole_derivatives"
polarizability_derivatives: Final[str] = "polarizability_derivatives"
total_charge: Final[str] = "total_charge"
partial_charges: Final[str] = "partial_charges"
spin_multiplicity: Final[str] = "spin_multiplicity"
electric_field: Final[str] = "electric_field"
magnetic_field: Final[str] = "magnetic_field"
nuclear_magnetic_moments: Final[str] = "nuclear_magnetic_moments"
shielding: Final[str] = "shielding"
nuclear_spin_coupling: Final[str] = "nuclear_spin_coupling"

## external fields needed for different response properties
required_external_fields = {
    dipole_moment: [electric_field],
    dipole_derivatives: [electric_field],
    partial_charges: [electric_field],
    polarizability: [electric_field],
    polarizability_derivatives: [electric_field],
    shielding: [magnetic_field],
    nuclear_spin_coupling: [magnetic_field],
}
