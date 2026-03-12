from typing import Dict, Tuple, Optional, Any

import torch
from tqdm import tqdm

import schnetpack.properties as properties
from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.loader import AtomsLoader

__all__ = ["calculate_stats", "estimate_atomrefs"]


def calculate_stats(
    dataset: ASEAtomsData,
    divide_by_atoms: Dict[str, bool],
    atomref: Dict[str, torch.Tensor] = None,
    batch_size: int = 10000,
    num_workers: int = 4,
    loader_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Use the incremental Welford algorithm described in [h1]_ to accumulate
    the mean and standard deviation over a set of samples.

    References:
    -----------
    .. [h1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        dataset: Dataset used to compute statistics.
        divide_by_atoms: Mapping from property name to bool indicating whether the property
            should be divided by the number of atoms before computing statistics.
        atomref: Optional single-atom reference values to subtract before computing statistics.
        batch_size: Batch size used for the temporary data loader.
        num_workers: Number of workers used by the data loader.

    Returns:
        Mapping from property name to `(mean, std)` tensors.
    """
    loader_kwargs = loader_kwargs or {}

    dataloader = AtomsLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **loader_kwargs,
    )

    property_names = list(divide_by_atoms.keys())
    norm_mask = torch.tensor(
        [float(divide_by_atoms[p]) for p in property_names], dtype=torch.float64
    )

    count = 0
    mean = torch.zeros_like(norm_mask)
    M2 = torch.zeros_like(norm_mask)

    for props in tqdm(dataloader, "calculating statistics"):
        sample_values = []
        for p in property_names:
            val = props[p][None, :]

            if atomref and p in atomref:
                ar = atomref[p]
                ar = ar[props[properties.Z]]
                idx_m = props[properties.idx_m]
                tmp = torch.zeros((idx_m[-1] + 1,), dtype=ar.dtype, device=ar.device)
                v0 = tmp.index_add(0, idx_m, ar)
                val -= v0

            sample_values.append(val)

        sample_values = torch.cat(sample_values, dim=0)

        batch_n = sample_values.shape[1]
        new_count = count + batch_n

        norm = norm_mask[:, None] * props[properties.n_atoms][None, :] + (
            1 - norm_mask[:, None]
        )
        sample_values = sample_values / norm

        sample_mean = torch.mean(sample_values, dim=1)
        sample_m2 = torch.sum((sample_values - sample_mean[:, None]) ** 2, dim=1)

        delta = sample_mean - mean
        mean += delta * batch_n / new_count
        corr = batch_n * count / new_count
        M2 += sample_m2 + delta**2 * corr
        count = new_count

    stddev = torch.sqrt(M2 / count)
    return {pn: (mu, std) for pn, mu, std in zip(property_names, mean, stddev)}


def estimate_atomrefs(
    dataset: ASEAtomsData,
    is_extensive: Dict[str, bool],
    z_max: int = 100,
    batch_size: int = 10000,
    num_workers: int = 4,
    loader_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Uses linear regression to estimate the elementwise biases (atomrefs).

    Args:
        dataset: Dataset used to estimate atom reference values.
        is_extensive: Mapping from property name to bool indicating whether the property is
            extensive. If False, atom type counts are divided by the number of atoms before fitting.
        z_max: Maximum atomic number used to size the atomref tensors.
        batch_size: Batch size used for the temporary data loader.
        num_workers: Number of workers used by the data loader.

    Returns:
        Mapping from property name to estimated atom reference tensor.
    """
    loader_kwargs = loader_kwargs or {}

    dataloader = AtomsLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **loader_kwargs,
    )

    property_names = list(is_extensive.keys())
    n_data = len(dataset)

    all_properties = {pname: torch.zeros(n_data) for pname in property_names}
    all_atom_types = torch.zeros((n_data, z_max))
    data_counter = 0

    for batch in tqdm(dataloader, "estimating atomrefs"):
        idx_m = batch[properties.idx_m]
        atomic_numbers = batch[properties.Z]

        for i in torch.unique(idx_m):
            atomic_numbers_i = atomic_numbers[idx_m == i]
            atom_types, atom_counts = torch.unique(atomic_numbers_i, return_counts=True)

            for atom_type, atom_count in zip(atom_types, atom_counts):
                all_atom_types[data_counter, atom_type] = atom_count

            for pname in property_names:
                property_value = batch[pname][i]
                if not is_extensive[pname]:
                    property_value *= batch[properties.n_atoms][i]
                all_properties[pname][data_counter] = property_value

            data_counter += 1

    existing_atom_types = torch.where(all_atom_types.sum(axis=0) != 0)[0]
    X = torch.squeeze(all_atom_types[:, existing_atom_types])

    weights = {}
    for pname in property_names:
        if is_extensive[pname]:
            weights[pname] = torch.linalg.inv(X.T @ X) @ X.T @ all_properties[pname]
        else:
            weights[pname] = (
                torch.linalg.inv(X.T @ X)
                @ X.T
                @ (all_properties[pname] / X.sum(axis=1))
            )

    out = {pname: torch.zeros((z_max,)) for pname in property_names}
    for pname in property_names:
        for atom_type, weight in zip(existing_atom_types, weights[pname]):
            out[pname][atom_type] = weight

    return out
