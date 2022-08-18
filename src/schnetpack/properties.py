"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
idx = "_idx"

## structure
Z = "_atomic_numbers"  #: nuclear charge
position = "_positions"  #: atom positions
R = position  #: atom positions

cell = "_cell"  #: unit cell
strain = "strain"
pbc = "_pbc"  #: periodic boundary conditions

seg_m = "_seg_m"  #: start indices of systems
idx_m = "_idx_m"  #: indices of systems
idx_i = "_idx_i"  #: indices of center atoms
idx_j = "_idx_j"  #: indices of neighboring atoms
idx_i_lr = "_idx_i_lr"  #: indices of center atoms for long-range
idx_j_lr = "_idx_j_lr"  #: indices of neighboring atoms for long-range

lidx_i = "_idx_i_local"  #: local indices of center atoms (within system)
lidx_j = "_idx_j_local"  #: local indices of neighboring atoms (within system)
Rij = "_Rij"  #: vectors pointing from center atoms to neighboring atoms
Rij_lr = (
    "_Rij_lr"  #: vectors pointing from center atoms to neighboring atoms for long range
)
n_atoms = "_n_atoms"  #: number of atoms
offsets = "_offsets"  #: cell offset vectors
offsets_lr = "_offsets_lr"  #: cell offset vectors for long range

R_strained = position + "_strained"  #: atom positions with strain-dependence
cell_strained = cell + "_strained"  #: atom positions with strain-dependence

n_nbh = "_n_nbh"  #: number of neighbors

#: indices of center atom triples
idx_i_triples = "_idx_i_triples"

#: indices of first neighboring atom triples
idx_j_triples = "_idx_j_triples"

#: indices of second neighboring atom triples
idx_k_triples = "_idx_k_triples"

## chemical properties
energy = "energy"
forces = "forces"
stress = "stress"
masses = "masses"
dipole_moment = "dipole_moment"
polarizability = "polarizability"
hessian = "hessian"
dipole_derivatives = "dipole_derivatives"
polarizability_derivatives = "polarizability_derivatives"
total_charge = "total_charge"
partial_charges = "partial_charges"
spin_multiplicity = "spin_multiplicity"
electric_field = "electric_field"
magnetic_field = "magnetic_field"
nuclear_magnetic_moments = "nuclear_magnetic_moments"
shielding = "shielding"
nuclear_spin_coupling = "nuclear_spin_coupling"

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
