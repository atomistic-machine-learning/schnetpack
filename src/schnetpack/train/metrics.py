import warnings

import numpy as np
import torch

from schnetpack import Properties


class Metric:
    r"""
    Base class for all metrics.

    Metrics measure the performance during the training and evaluation.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        self.target = target
        self.model_output = target if model_output is None else model_output
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.element_wise = element_wise

    def add_batch(self, batch, result):
        """ Add a batch to calculate the metric on """
        raise NotImplementedError

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        raise NotImplementedError

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        pass


class ModelBias(Metric):
    r"""
    Calculates the bias of the model. For non-scalar quantities, the mean of
    all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = "Bias_" + target if name is None else name
        super(ModelBias, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
            element_wise=element_wise,
        )

        self.l2loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l2loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        return y - yp

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1)).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += (
                torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
                * y.shape[-1]
            )
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class MeanSquaredError(Metric):
    r"""
    Metric for mean square error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "MSE_" + target if name is None else name
        super(MeanSquaredError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
            element_wise=element_wise,
        )

        self.bias_correction = bias_correction

        self.l2loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l2loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += (
                torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
                * y.shape[-1]
            )
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class RootMeanSquaredError(MeanSquaredError):
    r"""
    Metric for root mean square error. For non-scalar quantities, the mean of
    all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `RMSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "RMSE_" + target if name is None else name
        super(RootMeanSquaredError, self).__init__(
            target, model_output, bias_correction, name, element_wise=element_wise
        )

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return np.sqrt(self.l2loss / self.n_entries)


class MeanAbsoluteError(Metric):
    r"""
    Metric for mean absolute error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MAE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "MAE_" + target if name is None else name
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
            element_wise=element_wise,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
                    # print(result.shape)
            else:
                result = result[self.model_output]
            yp = result

        # print(yp, yp.shape, y.shape)
        diff = self._get_diff(y, yp)
        # print(diff)
        # print()
        self.l1loss += (
            torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        )
        if self.element_wise:
            self.n_entries += (
                torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
                * y.shape[-1]
            )
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries


class HeatmapMAE(MeanAbsoluteError):
    r"""
    Metric for heatmap of component-wise mean square error of non-scalar
    quantities.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `HeatmapMAE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = "HeatmapMAE_" + target if name is None else name
        super(HeatmapMAE, self).__init__(
            target, model_output, name=name, element_wise=element_wise
        )

    def add_batch(self, batch, result):
        if self.element_wise and torch.sum(batch[Properties.atom_mask] == 0) != 0:
            warnings.warn(
                "MAEHeatmap should not be used for element-wise "
                + "properties with different sized molecules!"
            )

        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l1loss += torch.sum(torch.abs(diff), 0).detach().cpu().data.numpy()
        self.n_entries += y.size(0)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries


class SumMAE(MeanAbsoluteError):
    r"""
    Metric for mean absolute error of length.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `LengthMAE_[target]` will be used (Default: None)
   """

    def __init__(
        self, target, model_output=None, axis=1, name=None, element_wise=False
    ):
        name = "SumMAE_" + target if name is None else name
        self.axis = axis
        super(SumMAE, self).__init__(
            target, model_output, name=name, element_wise=element_wise
        )

    def _get_diff(self, y, yp):
        ypl = torch.sum(yp, dim=self.axis)
        return torch.sum(torch.abs(y - ypl))


class LengthMSE(MeanSquaredError):
    r"""
    Metric for mean square error of length.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `LengthMSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = "LengthMSE_" + target if name is None else name
        super(LengthMSE, self).__init__(
            target, model_output, name=name, element_wise=element_wise
        )

    def _get_diff(self, y, yp):
        yl = torch.sqrt(torch.sum(y ** 2, dim=-1))
        ypl = torch.sqrt(torch.sum(yp ** 2, dim=-1))
        return torch.sum((yl - ypl) ** 2)


class LengthMAE(MeanAbsoluteError):
    r"""
    Metric for mean absolute error of length.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `LengthMAE_[target]` will be used (Default: None)
   """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = "LengthMAE_" + target if name is None else name
        super(LengthMAE, self).__init__(
            target, model_output, name=name, element_wise=element_wise
        )

    def _get_diff(self, y, yp):
        yl = torch.sqrt(torch.sum(y ** 2, dim=-1))
        ypl = torch.sqrt(torch.sum(yp ** 2, dim=-1))
        return torch.sum(torch.abs(yl - ypl))


class LengthRMSE(RootMeanSquaredError):
    r"""
    Metric for root mean square error of length.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `LengthRMSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
   """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = "LengthRMSE_" + target if name is None else name
        super(LengthRMSE, self).__init__(
            target, model_output, name=name, element_wise=element_wise
        )

    def _get_diff(self, y, yp):
        yl = torch.sqrt(torch.sum(y ** 2, dim=-1))
        ypl = torch.sqrt(torch.sum(yp ** 2, dim=-1))
        return torch.sum((yl - ypl) ** 2)


class AngleMSE(MeanSquaredError):
    r"""
    Metric for mean square error of angles.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `AngleMSE_[target]` will be used (Default: None)
    """

    def __init__(self, target, model_output=None, name=None):
        name = "AngleMSE_" + target if name is None else name
        super(AngleMSE, self).__init__(target, model_output, name=name)

    def _get_diff(self, y, yp):
        y = y / torch.norm(y, dim=1, keepdim=True)
        yp = yp / torch.norm(yp, dim=1, keepdim=True)

        diff = torch.matmul(
            y.view(y.size(0), 1, y.size(1)), yp.view(y.size(0), y.size(1), 1)
        )[:, 0]
        diff = torch.clamp(diff, -1, 1)
        angle = torch.acos(diff)

        return angle

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        y = y.view(-1, y.size(-1))
        yp = yp.view(-1, yp.size(-1))

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff ** 2).detach().cpu().data.numpy()
        self.n_entries += (
            torch.sum(torch.isnan(diff) == False).detach().cpu().data.numpy()
        )


class AngleMAE(MeanAbsoluteError):
    r"""
    Metric for mean absolute error of angles.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `AngleMAE_[target]` will be used (Default: None)
    """

    def __init__(self, target, model_output=None, name=None):
        name = "AngleMAE_" + target if name is None else name
        super(AngleMAE, self).__init__(target, model_output, name=name)

    def _get_diff(self, y, yp):
        y = y / torch.norm(y, dim=1, keepdim=True)
        yp = yp / torch.norm(yp, dim=1, keepdim=True)

        diff = torch.matmul(
            y.view(y.size(0), 1, y.size(1)), yp.view(y.size(0), y.size(1), 1)
        )[:, 0]
        diff = torch.clamp(diff, -1, 1)
        angle = torch.acos(diff)

        return angle

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        y = y.view(-1, y.size(-1))
        yp = yp.view(-1, yp.size(-1))

        diff = self._get_diff(y, yp)
        self.l1loss += torch.sum(torch.abs(diff)).detach().cpu().data.numpy()
        self.n_entries += (
            torch.sum(torch.isnan(diff) == False).detach().cpu().data.numpy()
        )


class AngleRMSE(RootMeanSquaredError):
    r"""
    Metric for root mean square error of angles.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `AngleRMSE_[target]` will be used (Default: None)
    """

    def __init__(self, target, model_output=None, name=None):
        name = "AngleRMSE_" + target if name is None else name
        super(AngleRMSE, self).__init__(target, model_output, name=name)

    def _get_diff(self, y, yp):
        y = y / torch.norm(y, dim=1, keepdim=True)
        yp = yp / torch.norm(yp, dim=1, keepdim=True)

        diff = torch.matmul(
            y.view(y.size(0), 1, y.size(1)), yp.view(y.size(0), y.size(1), 1)
        )
        diff = torch.clamp(diff, -1, 1)
        angle = torch.acos(diff) / float(np.pi) * 180.0

        return angle

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        y = y.view(-1, y.size(-1))
        yp = yp.view(-1, yp.size(-1))

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff ** 2).detach().cpu().data.numpy()
        self.n_entries += (
            torch.sum(torch.isnan(diff) == False).detach().cpu().data.numpy()
        )
