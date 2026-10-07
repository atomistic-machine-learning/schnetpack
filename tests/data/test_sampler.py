from types import SimpleNamespace

import torch

from schnetpack.data.sampler import ChunkedRandomSampler
from schnetpack.lightning.datamodule import AtomsDataModule


def _epochs(sampler, n_epochs):
    return [list(sampler) for _ in range(n_epochs)]


def test_chunked_random_sampler_len():
    sampler = ChunkedRandomSampler(data_source=range(10), num_samples=4)
    assert len(sampler) == 4
    assert len(list(sampler)) == 4


def test_chunked_random_sampler_covers_dataset_per_pass():
    torch.manual_seed(0)
    sampler = ChunkedRandomSampler(data_source=range(12), num_samples=4)

    epochs = _epochs(sampler, 3)
    assert sorted(i for epoch in epochs for i in epoch) == list(range(12))

    # the next pass is a fresh permutation, again covering the dataset
    epochs = _epochs(sampler, 3)
    assert sorted(i for epoch in epochs for i in epoch) == list(range(12))


def test_chunked_random_sampler_wraps_after_remainder():
    torch.manual_seed(0)
    sampler = ChunkedRandomSampler(data_source=range(10), num_samples=4)

    indices = [i for epoch in _epochs(sampler, 3) for i in epoch]
    # the first 10 indices cover the dataset before the permutation is redrawn
    assert sorted(indices[:10]) == list(range(10))


def test_setup_sampler_respects_num_samples_override():
    datamodule = SimpleNamespace(batch_size=2)
    batch_sampler = AtomsDataModule._setup_sampler(
        datamodule,
        sampler_cls=ChunkedRandomSampler,
        sampler_args={"num_samples": 4},
        dataset=range(10),
    )
    assert len(batch_sampler.sampler) == 4
    assert len(list(batch_sampler)) == 2
