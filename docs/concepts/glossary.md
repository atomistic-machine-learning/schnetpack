# Glossary

## Batch keys

A batch is a dictionary of tensors. Its keys are defined in
{mod}`schnetpack.properties`; the ones a model reads from every batch:

| Key | Name in `schnetpack.properties` | Content |
| --- | --- | --- |
| `_atomic_numbers` | `Z` | atomic number of every atom, shape `(n_atoms,)` |
| `_positions` | `R` | Cartesian positions, shape `(n_atoms, 3)` |
| `_n_atoms` | `n_atoms` | number of atoms of every structure, shape `(n_structures,)` |
| `_idx_m` | `idx_m` | index of the structure each atom belongs to, shape `(n_atoms,)` |
| `_cell` | `cell` | unit cell of every structure, shape `(n_structures, 3, 3)` |
| `_pbc` | `pbc` | periodic boundary conditions along the three cell vectors |
| `_idx_i`, `_idx_j` | `idx_i`, `idx_j` | center and neighbor atom of every pair in the neighbor list |
| `_offsets` | `offsets` | cell offset of every pair, for periodic structures |
| `_Rij` | `Rij` | vector from atom i to atom j of every pair |

Target properties use plain names: `energy`, `forces`, `stress`, `dipole_moment`
and so on. Keys that start with an underscore describe the structure; the others
are properties a model predicts or a loss compares.

## Terms

Atomwise output
: An output module that predicts a contribution per atom and sums or averages them
  per structure, e.g. {class}`schnetpack.model.Atomwise` for the energy.

Cutoff
: The distance beyond which two atoms do not interact. It sets which pairs
  enter the neighbor list and where the cutoff function brings each pair's
  contribution smoothly to zero.

Neighbor list
: All pairs of atoms closer than the cutoff, as the index arrays `idx_i` and
  `idx_j`. A transform such as
  {class}`schnetpack.transform.MatScipyNeighborList` computes it.

Representation
: The network that turns atomic numbers and positions into a feature vector per
  atom, e.g. {class}`schnetpack.model.SchNet`, {class}`schnetpack.model.PaiNN` or
  {class}`schnetpack.model.SO3net`.

Response property
: A property computed as a derivative of the model's energy, such as forces
  (minus the gradient with respect to the positions) or stress (from the derivative
  with respect to the strain).

Storage
: Where the structures of a dataset are held: an ASE database on disk or flat
  arrays in memory ({class}`schnetpack.data.ASEDBStorage`,
  {class}`schnetpack.data.ArrayStorage`).

Transform
: A module that changes a structure or a batch before or after the model, e.g.
  computing the neighbor list, removing an energy offset or casting to float32.
  See {mod}`schnetpack.transform`.
