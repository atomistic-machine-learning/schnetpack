from .fixtures import *
import torch.nn as nn
from numpy.testing import assert_array_equal


# class TestAtomisticModel:
#    def test_model_types(self, atomistic_model):
#        assert type(atomistic_model.output_modules) == nn.ModuleList
#
#    def test_forward_pass(self, atomistic_model, dataloader, result_shapes):
#        for batch in dataloader:
#            results = atomistic_model(batch)
#            for prop, result in results.items():
#                assert_array_equal(result.shape, result_shapes[prop])
