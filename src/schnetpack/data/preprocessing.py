import numpy as np


from sklearn.feature_extraction import DictVectorizer
from typing import Tuple


Array = np.ndarray


def get_per_atom_shift(z: Array, q: Array, pad_value: int = None) -> Tuple[Array, Array]:
    """
    Get per atom shift, given the atomic numbers across structures. The per atom shift is calculated by first constructing
    an matrix that counts the occurrences of each atomic type for each structure which gives a matrix `A` of shape
    (n_data, max_z+1), where `max_z` is the largest atomic number. The per atom shifts are then calculated by solving
    the linear equation `A @ x = q`. Here, `q` is the target quantity of shape (n_data). The function returns the
    shifts as a vector `shifts` of shape (max_z + 1), where e.g. the shift for carbon can be accessed by shifts[6], or
    for hydrogen by shifts[1]. The extra dimension comes from the fact that we want a one-to-one correspondence between
    index and atomic type. It also returns the rescaled quantities `q` using `shifts`. If one has differently sized
    structures, `z` has to be padded in order to solve the linear system,

    Args:
        z (Array): Atomic numbers, shape: (n_data, n)
        q (Array): Quantity to fit the shifts to, shape: (n_data)
        pad_value (int): If data has been padded, what is the value used for padding

    Returns: Tuple[Array]

    """
    u, _ = np.unique(z, return_counts=True)

    if pad_value is not None:
        idx_ = (u != pad_value)
    else:
        idx_ = np.arange(len(u))

    count_fn = lambda y: dict(zip(*np.unique(y, return_counts=True)))
    lhs_counts = list(map(count_fn, z))

    v = DictVectorizer(sparse=False)
    X = v.fit_transform(lhs_counts)
    X = X[..., idx_]

    sol = np.linalg.lstsq(X, q)

    shifts = np.zeros(np.max(u) + 1)
    for k, v in dict(zip(u[idx_], sol[0])).items():
        shifts[k] = v

    q_scaled = q - np.take(shifts, z).sum(axis=-1)
    return shifts, q_scaled
