from typing import Any

import torch
from tqdm import tqdm

import schnetpack.properties as properties
from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.loader import AtomsLoader

__all__ = ["calculate_stats", "estimate_atomrefs", "structure_stats", "fit_atomrefs"]


def _raw_view(dataset: ASEAtomsData, indices: list[int] | None) -> ASEAtomsData:
    """
    Statistics must always be computed on raw data. Build an index-restricted
    view of the dataset with no transforms and no split label, so that
    whatever view the caller holds (transformed, split-labeled) can never leak
    into statistics computation.
    """
    if indices is None:
        return dataset
    view = dataset.subset(list(indices))
    view.transforms = []
    view.split = None
    return view


def calculate_stats(
    dataset: ASEAtomsData,
    divide_by_atoms: dict[str, bool],
    atomref: dict[str, torch.Tensor] | None = None,
    batch_size: int = 10000,
    # num_workers=0: spawning workers by default breaks on platforms with
    # spawn start method (the dataset holds a DB connection), and stats are a
    # one-off pass anyway.
    num_workers: int = 0,
    loader_kwargs: dict[str, Any] | None = None,
    indices: list[int] | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
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
        indices: Optional indices into `dataset` to restrict the computation
            to. When given, statistics are computed on a raw view of these
            entries — any transforms or split label on `dataset` are ignored.

    Returns:
        Mapping from property name to `(mean, std)` tensors.
    """
    loader_kwargs = loader_kwargs or {}
    dataset = _raw_view(dataset, indices)

    dataloader = AtomsLoader(
        dataset,
        batch_size=batch_size,
        # Shuffling is irrelevant for mean/std over the full dataset; keep
        # iteration deterministic.
        shuffle=False,
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
    is_extensive: dict[str, bool],
    z_max: int = 100,
    batch_size: int = 10000,
    # num_workers=0: see calculate_stats — safe, deterministic defaults.
    num_workers: int = 0,
    loader_kwargs: dict[str, Any] | None = None,
    indices: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Uses linear regression to estimate the elementwise biases (atomrefs).

    Args:
        dataset: Dataset used to estimate atom reference values.
        is_extensive: Mapping from property name to bool indicating whether the property is
            extensive. If False, atom type counts are divided by the number of atoms before fitting.
        z_max: Maximum atomic number used to size the atomref tensors.
        batch_size: Batch size used for the temporary data loader.
        num_workers: Number of workers used by the data loader.
        indices: Optional indices into `dataset` to restrict the computation
            to. When given, atomrefs are estimated on a raw view of these
            entries — any transforms or split label on `dataset` are ignored.

    Returns:
        Mapping from property name to estimated atom reference tensor.
    """
    loader_kwargs = loader_kwargs or {}
    dataset = _raw_view(dataset, indices)

    dataloader = AtomsLoader(
        dataset,
        batch_size=batch_size,
        # Linear regression over all samples — order does not matter, keep
        # iteration deterministic.
        shuffle=False,
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


def _structure_values(y, n_atoms, owner, Z, divide_by_atoms, atomref) -> torch.Tensor:
    """A per-structure property in float64, without its atom references and per
    atom if asked: the values whose mean and standard deviation are the statistics."""
    y = y.double()
    if atomref is not None:
        ref = atomref.to(y.device, torch.float64)[Z]
        y = y - torch.zeros_like(y).index_add_(0, owner, ref)
    if divide_by_atoms:
        y = y / n_atoms
    return y


def _mean_std(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = y.mean()
    return mean.cpu(), ((y - mean) ** 2).mean().sqrt().cpu()


def _type_counts(
    owner: torch.Tensor, Z: torch.Tensor, z_max: int, n_structures: int
) -> torch.Tensor:
    """Atoms of each atomic number in each structure, (structures, z_max), float64."""
    counts = torch.zeros(n_structures, z_max, dtype=torch.float64, device=owner.device)
    return counts.index_put_((owner, Z), counts.new_ones(len(owner)), accumulate=True)


def _solve_atomrefs(XtX: torch.Tensor, Xty: torch.Tensor) -> torch.Tensor:
    """The least-squares references from the normal equations; 0 for absent elements."""
    present = XtX.diagonal() != 0
    out = torch.zeros_like(Xty)
    out[present] = torch.linalg.solve(XtX[present][:, present], Xty[present])
    return out.cpu()


def structure_stats(
    y: torch.Tensor,
    n_atoms: torch.Tensor,
    owner: torch.Tensor,
    Z: torch.Tensor,
    divide_by_atoms: bool,
    atomref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean and standard deviation of a per-structure property from flat tensors
    (float64, population standard deviation), on the tensors' device.
    :func:`calculate_stats` computes them for a dataset.

    Args:
        y: the property, one value per structure
        n_atoms: atoms per structure
        owner: the structure (0..) of each atom
        Z: the atomic number of each atom
        divide_by_atoms: per atom, divided by the atom count
        atomref: reference value per atomic number, subtracted first
    """
    return _mean_std(_structure_values(y, n_atoms, owner, Z, divide_by_atoms, atomref))


def fit_atomrefs(
    y: torch.Tensor,
    n_atoms: torch.Tensor,
    owner: torch.Tensor,
    Z: torch.Tensor,
    is_extensive: bool,
    z_max: int = 100,
    chunk: int = 100_000,
) -> torch.Tensor:
    """
    Atom references of a per-structure property by least squares over the
    atom-type counts (float64), on the tensors' device. :func:`estimate_atomrefs`
    fits them for a dataset. The normal equations are summed over chunks of
    structures, so the memory does not grow with the dataset.

    Args:
        y, n_atoms, owner, Z: as in :func:`structure_stats`; the atoms of a
            structure lie back to back (owner ascending)
        is_extensive: without effect, as in :func:`estimate_atomrefs`
        z_max: size of the result
        chunk: structures per chunk

    Returns:
        reference per atomic number, shape (z_max,); 0 for absent elements
    """
    y = y.double()
    XtX = torch.zeros(z_max, z_max, dtype=torch.float64, device=y.device)
    Xty = torch.zeros(z_max, dtype=torch.float64, device=y.device)
    atom_end = torch.cumsum(n_atoms, 0)
    for s in range(0, len(y), chunk):
        e = min(s + chunk, len(y))
        atoms = slice(int(atom_end[s] - n_atoms[s]), int(atom_end[e - 1]))
        X = _type_counts(owner[atoms] - s, Z[atoms], z_max, e - s)
        XtX += X.T @ X
        Xty += X.T @ y[s:e]
    return _solve_atomrefs(XtX, Xty)
