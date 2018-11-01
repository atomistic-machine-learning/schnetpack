import pytest
from torch.optim import Adam
import shutil
import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *
from schnetpack.loss_functions import MSELoss
from schnetpack.train.hooks import *
from schnetpack.metrics import *

@pytest.fixture
def data():
    return QM9("test_data/", properties=[QM9.U0])

@pytest.fixture
def data_splits(data):
    """
    train, val, test
    """
    return data.create_splits(20, 20)

@pytest.fixture
def train(data_splits):
    return data_splits[0]

@pytest.fixture
def val(data_splits):
    return data_splits[1]

@pytest.fixture
def test(data_splits):
    return data_splits[2]

@pytest.fixture
def loader(train):
    return spk.data.AtomsLoader(train, batch_size=10, num_workers=4)

@pytest.fixture
def val_loader(val):
    return spk.data.AtomsLoader(val)

@pytest.fixture
def reps():
    return rep.SchNet()

@pytest.fixture
def output():
    return atm.Atomwise()

@pytest.fixture
def model(reps, output):
    return atm.AtomisticModel(reps, output)

@pytest.fixture
def loss():
    return MSELoss(input_key=QM9.U0, target_key='y')

@pytest.fixture
def max_epoch_hook():
    return MaxEpochHook(2)

@pytest.fixture
def mae():
    return MeanAbsoluteError(QM9.U0, 'y')

@pytest.fixture
def csv_hook(mae):
    return CSVHook('log/', [mae])

@pytest.fixture
def trainer(model, loss, max_epoch_hook, csv_hook):
    return spk.train.Trainer("output/", model, loss, Adam,
                             optimizer_params=dict(lr=1e-4),
                             hooks=[max_epoch_hook, csv_hook])

@pytest.fixture
def cfg_path():
    return 'tmp_config_file'


class TestQM9Script:

    def test_train(self, trainer, loader, val_loader):
        trainer.train(torch.device("cpu"), loader, val_loader)
        log_array = np.loadtxt('log/log.csv', delimiter=",", skiprows=1)
        assert log_array.shape[0] == 2


    def test_dump_load_train(self, trainer, loader, val_loader, cfg_path):
        trainer.dump_config(cfg_path)
        restored_trainer = spk.train.Trainer.from_json(cfg_path)
        restored_trainer.train(torch.device("cpu"), loader, val_loader)
        log_array = np.loadtxt('log/log.csv', delimiter=",", skiprows=1)
        assert log_array.shape[0] == 2

    def teardown_method(self):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists(cfg_path()):
            os.remove(cfg_path())
        if os.path.exists('output/'):
            shutil.rmtree('output/')
        if os.path.exists('log/'):
            shutil.rmtree('log/')