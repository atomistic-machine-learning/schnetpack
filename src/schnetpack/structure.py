"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility
"""
from typing import Final

idx: Final[str] = "_idx"
Z: Final[str] = "_atomic_numbers"
position: Final[str] = "_positions"
R: Final[str] = position

cell: Final[str] = "_cell"
cell_offset: Final[str] = "_cell_offset"
pbc: Final[str] = "_pbc"

idx_m: Final[str] = "_seg_m"
idx_i: Final[str] = "_idx_i"
idx_j: Final[str] = "_idx_j"
Rij: Final[str] = "_Rij"
n_atoms: Final[str] = "_n_atoms"

idx_i_triples: Final[str] = "_idx_i_triples"
idx_j_triples: Final[str] = "_idx_j_triples"
idx_k_triples: Final[str] = "_idx_k_triples"
