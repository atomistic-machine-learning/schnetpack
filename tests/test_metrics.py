import pytest
from schnetpack.train.metrics import *
from schnetpack.train.metrics import (
    ModelBias,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    HeatmapMAE,
    AngleMAE,
)
from schnetpack.train import build_mse_loss


@pytest.fixture
def properties():
    return ["_energy", "_forces", "_dipole_moment"]


@pytest.fixture
def batch():
    return dict(
        _atom_mask=torch.DoubleTensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        _forces=torch.DoubleTensor(
            [
                [[8, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[1, 1, 0], [1, 4, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        _energy=torch.DoubleTensor([[1], [1]]),
        _dipole_moment=torch.DoubleTensor([[1, 2, 3], [4, 5, 6]]),
    )


@pytest.fixture
def result():
    return dict(
        dydx=torch.DoubleTensor(
            [
                [[8, 1, 1], [0, 2, 1], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 3], [0, 1, 4], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        y=torch.DoubleTensor([[2], [2]]),
        _dipole_moment=torch.DoubleTensor([[0, 2, 0], [4, 1, 1]]),
    )


@pytest.fixture
def result_named():
    return dict(
        _forces=torch.DoubleTensor(
            [
                [[8, 1, 1], [0, 2, 1], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 3], [0, 1, 4], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        _energy=torch.DoubleTensor([[2], [2]]),
        _dipole_moment=torch.DoubleTensor([[0, 2, 0], [4, 1, 1]]),
    )


@pytest.fixture
def diff_named(batch, result_named):
    return dict(
        _forces=batch["_forces"] - result_named["_forces"],
        _energy=batch["_energy"] - result_named["_energy"],
        _dipole_moment=batch["_dipole_moment"] - result_named["_dipole_moment"],
    )


@pytest.fixture
def loss_tradeoff():
    return [1.0, 1.0, 0.0]


@pytest.fixture
def loss_value(diff_named):
    return sum([diff.pow(2).mean() for diff in diff_named.values()])


@pytest.fixture
def loss_value_traded(diff_named):
    return sum(
        [diff_named["_energy"].pow(2).mean(), diff_named["_forces"].pow(2).mean()]
    )


@pytest.fixture
def diff(batch, result):
    return dict(
        dydx=batch["_forces"] - result["dydx"], y=batch["_energy"] - result["y"]
    )


@pytest.fixture
def mse_result(diff):
    return dict(
        dydx=torch.sum(diff["dydx"] ** 2).detach().cpu().data.numpy() / 15,
        y=torch.sum(diff["y"] ** 2).detach().cpu().data.numpy() / 2,
    )


@pytest.fixture
def mae_result(diff):
    return dict(
        dydx=torch.sum(torch.abs(diff["dydx"])).detach().cpu().data.numpy() / 15,
        y=torch.sum(torch.abs(diff["y"])).detach().cpu().data.numpy() / 2,
    )


@pytest.fixture
def rmse_result(mse_result):
    returns = {}
    for key, value in mse_result.items():
        returns[key] = np.sqrt(value)
    return returns


@pytest.fixture
def bias_result(diff):
    return dict(
        dydx=torch.sum(diff["dydx"]).detach().cpu().data.numpy() / 15,
        y=torch.sum(diff["y"]).detach().cpu().data.numpy() / 2,
    )


@pytest.fixture
def heatmap_mae_result(diff):
    return dict(
        dydx=torch.sum(torch.abs(diff["dydx"]), 0).detach().cpu().data.numpy() / 2,
        y=torch.sum(torch.abs(diff["y"])).detach().cpu().data.numpy() / 2,
    )


@pytest.fixture
def energy_mse():
    return MeanSquaredError("_energy", "y", name="energy")


@pytest.fixture
def forces_mse():
    return MeanSquaredError("_forces", "dydx", name="forces", element_wise=True)


@pytest.fixture
def energy_rmse():
    return RootMeanSquaredError("_energy", "y", name="energy")


@pytest.fixture
def forces_rmse():
    return RootMeanSquaredError("_forces", "dydx", name="forces", element_wise=True)


@pytest.fixture
def energy_bias():
    return ModelBias("_energy", "y", name="energy")


@pytest.fixture
def forces_bias():
    return ModelBias("_forces", "dydx", name="forces", element_wise=True)


@pytest.fixture
def energy_mae():
    return MeanAbsoluteError("_energy", "y", name="energy")


@pytest.fixture
def forces_mae():
    return MeanAbsoluteError("_forces", "dydx", name="forces", element_wise=True)


@pytest.fixture
def energy_heatmapmae():
    return HeatmapMAE("_energy", "y")


@pytest.fixture
def forces_heatmapmae():
    return HeatmapMAE("_forces", "dydx", element_wise=True)


@pytest.fixture
def dipole_angle_mae():
    return AngleMAE("_dipole_moment", "_dipole_moment")


@pytest.fixture
def forces_angle_mae():
    return AngleMAE("_forces", "dydx")


class TestMetrics:
    def test_mae(self):
        pass

    def assert_valid_metric(self, metric, batch, result, target):
        metric.add_batch(batch, result)
        metric.add_batch(batch, result)
        m = metric.aggregate()
        assert np.equal(m, target).all()
        if hasattr(m, "__iter__"):
            assert len(m.shape) != 0

    def test_energy_mse(self, energy_mse, batch, result, mse_result):
        val_metric = mse_result["y"]
        self.assert_valid_metric(energy_mse, batch, result, val_metric)

    def test_forces_mse(self, forces_mse, batch, result, mse_result):
        val_metric = mse_result["dydx"]
        self.assert_valid_metric(forces_mse, batch, result, val_metric)

    def test_energy_rmse(self, energy_rmse, batch, result, rmse_result):
        val_metric = rmse_result["y"]
        self.assert_valid_metric(energy_rmse, batch, result, val_metric)

    def test_forces_rmse(self, forces_rmse, batch, result, rmse_result):
        val_metric = rmse_result["dydx"]
        self.assert_valid_metric(forces_rmse, batch, result, val_metric)

    def test_energy_bias(self, energy_bias, batch, result, bias_result):
        val_metric = bias_result["y"]
        self.assert_valid_metric(energy_bias, batch, result, val_metric)

    def test_forces_bias(self, forces_bias, batch, result, bias_result):
        val_metric = bias_result["dydx"]
        self.assert_valid_metric(forces_bias, batch, result, val_metric)

    def test_energy_mea(self, energy_mae, batch, result, mae_result):
        val_metric = mae_result["y"]
        self.assert_valid_metric(energy_mae, batch, result, val_metric)

    def test_forces_mea(self, forces_mae, batch, result, mae_result):
        val_metric = mae_result["dydx"]
        self.assert_valid_metric(forces_mae, batch, result, val_metric)

    def test_enegry_heatmapmae(
        self, energy_heatmapmae, batch, result, heatmap_mae_result
    ):
        val_metric = heatmap_mae_result["y"]
        self.assert_valid_metric(energy_heatmapmae, batch, result, val_metric)

    def test_forces_heatmapmae(
        self, forces_heatmapmae, batch, result, heatmap_mae_result
    ):
        val_metric = heatmap_mae_result["dydx"]
        with pytest.warns(UserWarning):
            self.assert_valid_metric(forces_heatmapmae, batch, result, val_metric)

    def test_angle_entries(self, forces_angle_mae, dipole_angle_mae, batch, result):
        forces_angle_mae.add_batch(batch, result)
        n_entries = forces_angle_mae.n_entries
        assert np.equal(n_entries, 5)
        if hasattr(n_entries, "__iter__"):
            assert len(n_entries.shape) != 0
        dipole_angle_mae.add_batch(batch, result)
        n_entries = dipole_angle_mae.n_entries
        assert np.equal(n_entries, 2)
        if hasattr(n_entries, "__iter__"):
            assert len(n_entries.shape) != 0

    def test_loss(self, batch, result_named, properties, loss_value):
        loss_fn = build_mse_loss(properties)
        loss = loss_fn(batch, result_named)
        assert np.equal(loss, loss_value)

    def test_loss_tradeoff(
        self, batch, result_named, properties, loss_value_traded, loss_tradeoff
    ):
        loss_fn = build_mse_loss(properties, loss_tradeoff)
        loss = loss_fn(batch, result_named)
        assert np.equal(loss, loss_value_traded)
