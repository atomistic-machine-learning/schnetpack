import logging
import os

import pytest
import torch

import schnetpack.properties as structure
from schnetpack.transform import ASENeighborList, CachedNeighborList


class CountingNeighborList(ASENeighborList):
    def __init__(self, cutoff):
        super().__init__(cutoff)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return super().forward(inputs)


@pytest.fixture
def cached_nbl(tmp_path, cutoff):
    return CachedNeighborList(
        cache_path=str(tmp_path / "cache"),
        neighbor_list=CountingNeighborList(cutoff),
    )


@pytest.fixture
def inputs(four_atoms):
    return {**four_atoms, structure.idx: torch.tensor([0])}


def assert_same_neighbors(a, b):
    for key in (structure.idx_i, structure.idx_j, structure.offsets):
        torch.testing.assert_close(a[key], b[key])


def test_cache_is_written_once_and_reused(cached_nbl, inputs):
    first = cached_nbl(dict(inputs))
    second = cached_nbl(dict(inputs))

    assert cached_nbl.neighbor_list.calls == 1
    assert_same_neighbors(first, second)
    assert sorted(os.listdir(cached_nbl.cache_location)) == [
        "cache_0.lock",
        "cache_0.pt",
    ]


def test_corrupt_cache_is_rebuilt(cached_nbl, inputs, caplog):
    reference = cached_nbl(dict(inputs))
    cache_file = os.path.join(cached_nbl.cache_location, "cache_0.pt")
    with open(cache_file, "wb") as f:
        f.write(b"truncated")

    with caplog.at_level(logging.WARNING, logger="schnetpack.transform.neighborlist"):
        rebuilt = cached_nbl(dict(inputs))

    assert cached_nbl.neighbor_list.calls == 2
    assert_same_neighbors(rebuilt, reference)
    assert "rebuilding it" in caplog.text
    # the rebuilt cache replaced the corrupt file
    assert_same_neighbors(torch.load(cache_file, weights_only=True), reference)
