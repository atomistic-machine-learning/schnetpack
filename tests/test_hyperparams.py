import pytest
import os
from schnetpack.config_model import Hyperparameters


class Dummy(object):
    def __init__(self, dummy_a):
        self.dummy_a = dummy_a

@pytest.fixture()
def sub_default_config():
    return dict(s1=1, s2=2, s3=3)

class SubNetwork(Hyperparameters):
    default_parameters = sub_default_config()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __eq__(self, other):
        for attr in ['s1', 's2', 's3']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

subnetwork = SubNetwork(**sub_default_config())

@pytest.fixture()
def default_config():
    return dict(a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork)

@pytest.fixture()
def config():
    return dict(c=13, d=14, e=15, dummy=Dummy)

@pytest.fixture()
def updated_config():
    return dict(a=1, b=2, c=13, d=14, dummy=Dummy, sub=subnetwork)

class Network(Hyperparameters):
    default_parameters = default_config()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __eq__(self, other):
        for attr in ['a', 'b', 'c', 'd', 'dummy']:
            if getattr(self, attr) != getattr(other, attr):
                return False
            if self.sub != other.sub:
                return False
        return True


@pytest.fixture()
def network():
    return Network



class TestHyperparams(object):

    def test_init(self, network, config):
        net = network(**config)
        assert net.a == 1
        assert net.b == 2
        assert net.c == 13
        assert net.d == 14

    def test_config_update(self, network, config, updated_config):
        net = network(**config)
        assert net.config == updated_config

    def test_empty_config_update(self, network, default_config):
        net = network()
        assert net.config == default_config

    def test_empty_config_init(self, network, default_config):
        net = network()
        for key in default_config.keys():
            assert getattr(net, key) == default_config[key]

    def test_dump_and_load(self, network, config):
        net = network(**config)
        net.dump_config('config_file')
        net2 = network().from_json('config_file')
        for attribute in ['a', 'b', 'c', 'd', 'config', 'default_parameters', 'dummy']:
            assert getattr(net, attribute) == getattr(net2, attribute)

    @classmethod
    def teardown_class(cls):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('config_file'):
            os.remove('config_file')





