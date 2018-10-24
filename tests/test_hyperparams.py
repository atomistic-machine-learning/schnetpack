import os
from schnetpack.config_model import Hyperparameters
from schnetpack.representation.schnet import SchNet

class Dummy(object):
    def __init__(self, dummy_a):
        self.dummy_a = dummy_a

class SubNetwork(Hyperparameters):

    def __init__(self, s1=1, s2=2, s3=3):
        Hyperparameters.__init__(self, locals())
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def __eq__(self, other):
        for attr in ['s1', 's2', 's3']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

subnetwork = SubNetwork()

class Network(Hyperparameters):

    def __init__(self, a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork):
        Hyperparameters.__init__(self, locals())
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dummy = dummy
        self.sub = sub

    def __eq__(self, other):
        for attr in ['a', 'b', 'c', 'd', 'dummy']:
            if getattr(self, attr) != getattr(other, attr):
                return False
            if self.sub != other.sub:
                return False
        return True

net = Network(c=13, d=14)
default_net = Network()
default_config = dict(a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork)
updated_config = dict(a=1, b=2, c=13, d=14, dummy=Dummy, sub=subnetwork)

class TestHyperparams(object):

    def test_init(self):
        assert net.a == 1
        assert net.b == 2
        assert net.c == 13
        assert net.d == 14

    def test_config_update(self):
        assert net.config == updated_config

    def test_empty_config_update(self):
        assert default_net.config == default_config

    def test_empty_config_init(self):
        for key in default_config.keys():
            assert getattr(default_net, key) == default_config[key]

    def test_dump_and_load(self):
        net.dump_config('config_file')
        net2 = Network().from_json('config_file')
        for attribute in ['a', 'b', 'c', 'd', 'config', 'default_config', 'dummy']:
            assert getattr(net, attribute) == getattr(net2, attribute)

    @classmethod
    def teardown_class(cls):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('config_file'):
            os.remove('config_file')

class TestSchnetHyperparams(object):

    def_schnet = SchNet()
    mod_schnet = SchNet(n_gaussians=999, n_filters=999)

    def test_dump_and_load(self):
        self.def_schnet.dump_config('def_schnet_config')
        loaded_schnet = SchNet().from_json('def_schnet_config')
        assert self.def_schnet.config == loaded_schnet.config

    @classmethod
    def teardown_class(cls):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('def_schnet_config'):
            os.remove('def_schnet_config')

