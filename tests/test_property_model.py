from .fixtures import *
import torch.nn as nn
from numpy.testing import assert_array_equal


class TestPropertyModel:

    def test_model_types(self, property_model):
        assert type(property_model.output_modules) == nn.ModuleList

    def test_forward_pass(self, new_atomistic_model, dataloader, properties,
                          result_shapes):
        for batch in dataloader:
            results = new_atomistic_model(batch)
            for prop, result in results.items():
                assert_array_equal(result.shape, result_shapes[prop])

