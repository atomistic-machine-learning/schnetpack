"""
Classes for constructing the different representations available in SchnetPack. This encompasses SchNet [#schnet4]_,
Behler-type atom centered symmetry functions (ACSF) [#acsf2]_ and a weighted variant thereof (wACSF) [#wacsf2]_.

References
----------
.. [#schnet4] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
   Quantum-chemical insights from deep tensor neural networks.
   Nature Communications, 8, 13890. 2017.
.. [#acsf2] Behler:
   Atom-centered symmetry functions for constructing high-dimensional neural network potentials.
   The Journal of Chemical Physics 134. 074106. 2011.
.. [#wacsf2] Gastegger, Schwiedrzik, Bittermann, Berzsenyi, Marquetand:
   wACSF -- Weighted atom-centered symmetry functions as descriptors in machine learning potentials.
   The Journal of Chemical Physics 148 (24), 241709. 2018.
"""

from schnetpack.representation.base import *
from schnetpack.representation.schnet import *
from schnetpack.representation.hdnn import *
from schnetpack.representation.physnet import *


def update_model_to_latest_version(model):
    import torch.nn as nn
    import schnetpack as spk

    # check if model is deprecated
    if not model_deprecated(model):
        print("The model is already up to date!")
        return model

    # rename distances to distance_provider
    if not hasattr(model.representation, "distance_provider"):
        model.representation.distance_provider = model.representation.distances
        del model.representation.distances

    # add post_interactions
    if not hasattr(model.representation, "post_interactions"):
        model.representation.post_interactions = nn.ModuleList(
            nn.Identity() for _ in range(len(model.representation.interactions))
        )

    # add interaction aggregation
    if not hasattr(model.representation, "interaction_aggregation"):
        model.representation.interaction_aggregation = spk.representation.InteractionAggregation(
            "last"
        )

    # add return_distances argument
    if not hasattr(model.representation, "return_distances"):
        model.representation.return_distances = False

    # add pre_activation to Dense layers
    for module in model.modules():
        if type(module) == spk.nn.Dense:
            module.pre_activation = nn.Identity()
            if module.activation is None:
                module.activation = nn.Identity()
    return model


def update_saved_model_to_latest_version(model_path):
    import torch

    model = torch.load(model_path)
    model = update_model_to_latest_version(model)
    torch.save(model, model_path)


def model_deprecated(model):
    import schnetpack as spk

    if not hasattr(model.representation, "distance_provider"):
        return True
    if not hasattr(model.representation, "post_interactions"):
        return True
    if not hasattr(model.representation, "interaction_aggregation"):
        return True
    if not hasattr(model.representation, "return_distances"):
        return True
    for module in model.modules():
        if type(module) == spk.nn.Dense:
            if not hasattr(module, "pre_activation"):
                return True
            if module.activation is None:
                return True
    return False


def clone(module, n_clones, shared_weights=False):
    import torch.nn as nn
    import copy

    # return modules with shared weights
    if shared_weights:
        return nn.ModuleList([module for _ in range(n_clones)])

    # return modules without shared weights
    modules = nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])
    for new_module in modules:
        new_module.reset_parameters()

    return modules
