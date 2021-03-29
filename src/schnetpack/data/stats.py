from typing import Dict, Tuple
import torch
from torch_scatter import segment_sum_csr

from schnetpack import Structure
from schnetpack.data import AtomsLoader
from tqdm import tqdm


def calculate_stats(
    dataloader: AtomsLoader,
    normalize_prop_by_atoms: Dict[str, bool],
    atomref: Dict[str, torch.Tensor] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Use the incremental Welford algorithm described in [1]_ to accumulate
    the mean and standard deviation over a set of samples.

    References:
    -----------
    .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        dataset: atoms data set
        normalize_prop_by_atoms: dict from property name to bool:
            If True, divide property by number of atoms before calculating statistics.
        atomref: reference values for single atoms to be removed before calculating stats


    Returns:

    """
    property_names = list(normalize_prop_by_atoms.keys())
    norm_mask = torch.tensor(
        [float(normalize_prop_by_atoms[p]) for p in property_names], dtype=torch.float64
    )

    count = 0
    mean = torch.zeros_like(norm_mask)
    M2 = torch.zeros_like(norm_mask)
    atomref = {k: torch.tensor(v) for k, v in atomref.items()}

    for props in tqdm(dataloader):
        sample_values = []
        for p in property_names:
            val = props[p][None, :]
            if atomref and p in atomref.keys():
                ar = atomref[p]
                ar = ar[props[Structure.Z]]
                v0 = segment_sum_csr(ar, props[Structure.seg_m])
                val -= v0

            sample_values.append(val)
        sample_values = torch.cat(sample_values, dim=0)

        batch_size = sample_values.shape[1]
        new_count = count + batch_size

        norm = norm_mask[:, None] * props[Structure.n_atoms][None, :] + (
            1 - norm_mask[:, None]
        )
        sample_values /= norm

        sample_mean = torch.mean(sample_values, dim=1)
        sample_m2 = torch.sum((sample_values - sample_mean[:, None]) ** 2, dim=1)

        delta = sample_mean - mean
        mean += delta * batch_size / new_count
        corr = batch_size * count / new_count
        M2 += sample_m2 + delta ** 2 * corr
        count = new_count

    stddev = torch.sqrt(M2 / count)
    stats = {pn: (mu, std) for pn, mu, std in zip(property_names, mean, stddev)}
    return stats
