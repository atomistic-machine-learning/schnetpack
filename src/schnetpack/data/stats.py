from typing import Dict, Tuple

import torch
from tqdm import tqdm

import schnetpack.properties as properties
from schnetpack.data import AtomsLoader

__all__ = ["calculate_stats", "estimate_atomrefs"]


def calculate_stats(
    dataloader: AtomsLoader,
    divide_by_atoms: Dict[str, bool],
    atomref: Dict[str, torch.Tensor] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Use the incremental Welford algorithm described in [h1]_ to accumulate
    the mean and standard deviation over a set of samples.

    References:
    -----------
    .. [h1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        dataloader: data loader
        divide_by_atoms: dict from property name to bool:
            If True, divide property by number of atoms before calculating statistics.
        atomref: reference values for single atoms to be removed before calculating stats

    Returns:
        Mean and standard deviation over all samples

    """
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
            if atomref and p in atomref.keys():
                ar = atomref[p]
                ar = ar[props[properties.Z]]
                idx_m = props[properties.idx_m]
                tmp = torch.zeros((idx_m[-1] + 1,), dtype=ar.dtype, device=ar.device)
                v0 = tmp.index_add(0, idx_m, ar)
                val -= v0

            sample_values.append(val)
        sample_values = torch.cat(sample_values, dim=0)

        batch_size = sample_values.shape[1]
        new_count = count + batch_size

        norm = norm_mask[:, None] * props[properties.n_atoms][None, :] + (
            1 - norm_mask[:, None]
        )
        sample_values /= norm

        sample_mean = torch.mean(sample_values, dim=1)
        sample_m2 = torch.sum((sample_values - sample_mean[:, None]) ** 2, dim=1)

        delta = sample_mean - mean
        mean += delta * batch_size / new_count
        corr = batch_size * count / new_count
        M2 += sample_m2 + delta**2 * corr
        count = new_count

    stddev = torch.sqrt(M2 / count)
    stats = {pn: (mu, std) for pn, mu, std in zip(property_names, mean, stddev)}
    return stats


def estimate_atomrefs(dataloader, is_extensive, z_max=100):
    """
    Uses linear regression to estimate the elementwise biases (atomrefs).

    Args:
        dataloader: data loader
        is_extensive: If True, divide atom type counts by number of atoms before
            calculating statistics.

    Returns:
        Elementwise bias estimates over all samples

    """
    property_names = list(is_extensive.keys())
    n_data = len(dataloader.dataset)
    all_properties = {pname: torch.zeros(n_data) for pname in property_names}
    all_atom_types = torch.zeros((n_data, z_max))
    data_counter = 0

    # loop over all batches
    for batch in tqdm(dataloader, "estimating atomrefs"):
        # load data
        idx_m = batch[properties.idx_m]
        atomic_numbers = batch[properties.Z]

        # get counts for atomic numbers
        unique_ids = torch.unique(idx_m)
        for i in unique_ids:
            atomic_numbers_i = atomic_numbers[idx_m == i]
            atom_types, atom_counts = torch.unique(atomic_numbers_i, return_counts=True)
            # save atom counts and properties
            for atom_type, atom_count in zip(atom_types, atom_counts):
                all_atom_types[data_counter, atom_type] = atom_count
            for pname in property_names:
                property_value = batch[pname][i]
                if not is_extensive[pname]:
                    property_value *= batch[properties.n_atoms][i]
                all_properties[pname][data_counter] = property_value
            data_counter += 1

    # perform linear regression to get the elementwise energy contributions
    existing_atom_types = torch.where(all_atom_types.sum(axis=0) != 0)[0]
    X = torch.squeeze(all_atom_types[:, existing_atom_types])
    w = dict()
    for pname in property_names:
        if is_extensive[pname]:
            w[pname] = torch.linalg.inv(X.T @ X) @ X.T @ all_properties[pname]
        else:
            w[pname] = (
                torch.linalg.inv(X.T @ X)
                @ X.T
                @ (all_properties[pname] / X.sum(axis=1))
            )

    # compute energy estimates
    elementwise_contributions = {
        pname: torch.zeros((z_max)) for pname in property_names
    }
    for pname in property_names:
        for atom_type, weight in zip(existing_atom_types, w[pname]):
            elementwise_contributions[pname][atom_type] = weight

    return elementwise_contributions
