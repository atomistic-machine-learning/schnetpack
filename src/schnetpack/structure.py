"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from typing import Final


idx: Final[str] = "_idx"

Z: Final[str] = "_atomic_numbers"  #: nuclear charge
position: Final[str] = "_positions"  #: atom positions
R: Final[str] = position  #: atom positions

cell: Final[str] = "_cell"  #: unit cell
pbc: Final[str] = "_pbc"  #: periodic boundary conditions

idx_m: Final[str] = "_idx_m"  #: indices of systems
idx_i: Final[str] = "_idx_i"  #: indices of center atoms
idx_j: Final[str] = "_idx_j"  #: indices of neighboring atoms
Rij: Final[str] = "_Rij"  #: indices of atom pairs
n_atoms: Final[str] = "_n_atoms"  #: number of atoms
