from sacred import Ingredient
from schnetpack.metrics import *

metrics_ing = Ingredient('metrics')


@metrics_ing.config
def config():
    r"""
    Settings for metrics that will be used for logging the training session.
    """
    element_wise = ['forces']   # list of elementwise properties (e.g. forces)


@metrics_ing.capture
def build_metrics(names, property_map, element_wise):
    """
    builds a list with schnetpack.metrics.Metric objects

    Args:
        names (list): names of the metrics that should be used
        property_map (dict): mapping between model properties and dataset
            properties
        element_wise (list): list of the element_wise properties

    Returns:
        list of schnetpack.metrics.Metric objects
    """
    metrics_objects = []
    for metric in names:
        metric = metric.lower()
        if metric == 'mae':
            metrics_objects +=\
                [MeanAbsoluteError(tgt, p, element_wise=p in element_wise)
                 for p, tgt in property_map.items() if tgt is not None]
        elif metric == 'rmse':
            metrics_objects +=\
                [RootMeanSquaredError(tgt, p, element_wise=p in element_wise)
                 for p, tgt in property_map.items() if tgt is not None]
        else:
            raise NotImplementedError
    return metrics_objects
