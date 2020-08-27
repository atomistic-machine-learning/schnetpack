import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
from schnetpack import Properties

__all__ = ["InteractionAggregation", "AtomisticRepresentation", "InteractionRefinement"]


class InteractionAggregation(nn.Module):
    def __init__(self, mode="sum"):
        # check if interaction mode is valid
        if not mode in ["sum", "mean", "last"]:
            raise NotImplementedError(
                "The selected aggregation mode is not implemented!"
            )

        # call super
        super(InteractionAggregation, self).__init__()

        # save attributes
        self.mode = mode

    def forward(self, intermediate_interactions, x):
        # return x + v of last interaction layer
        if self.mode == "last":
            return x

        # stack inputs, if list type
        if type(intermediate_interactions) == list:
            inputs = torch.stack(intermediate_interactions)

        # representation by aggregating intermediate interactions
        if self.mode == "sum":
            return torch.sum(inputs, axis=0)
        if self.mode == "mean":
            return torch.mean(inputs, axis=0)

        # raise error if no valid aggregation_mode is selected
        raise NotImplementedError(
            "Aggregation mode {} is not implemented!".format(self.mode)
        )


class InteractionRefinement(nn.Module):
    def __init__(self, n_features, property):
        # initialize layer
        super(InteractionRefinement, self).__init__()

        # attributes
        self.property = property

        # trainable parameters
        self.register_parameter('embedding', nn.Parameter(torch.Tensor(n_features)))
        self.register_parameter('keys', nn.Parameter(torch.Tensor(n_features)))

    def _attention_weights(self, x, property, atom_mask):
        # compute weights
        w = F.softplus(torch.sum(x * (property.sign() * self.keys).unsqueeze(1), -1))
        w = w * atom_mask
        # compute weight norms
        wsum = w.sum(-1, keepdim=True)

        # return normalized weights; 1e-8 prevents possible division by 0
        return w / (wsum + 1e-8)

    def reset_parameters(self):
        nn.init.zeros_(self.embedding)
        nn.init.zeros_(self.keys)

    def forward(self, inputs):
        # get input data
        x = inputs["x"]
        additional_feature = inputs[self.property]
        atom_mask = inputs[Properties.atom_mask]

        # compute attention weights
        attention_weights = self._attention_weights(x, additional_feature, atom_mask)

        # compute refinement tensors
        refinement_layer = (additional_feature * attention_weights).unsqueeze(
            -1
        ) * self.embedding

        return refinement_layer


class AtomisticRepresentation(nn.Module):
    def __init__(
        self,
        embedding,
        distance_expansion,
        interactions,
        pre_interactions=None,
        interaction_refinements=None,
        interaction_outputs=None,
        post_interactions=None,
        interaction_aggregation=InteractionAggregation(mode="last"),
        return_intermediate=False,
        return_distances=False,
        sum_before_interaction_append=False,
    ):
        # initialize layer
        super(AtomisticRepresentation, self).__init__()

        # set version tag
        self.version = spk.__version__

        # attributes
        self.return_intermediate = return_intermediate
        self.return_distances = return_distances
        self.sum_before_interaction_append = sum_before_interaction_append

        # modules
        # embedding
        self.embedding = embedding

        # distances
        self.distance_provider = spk.nn.AtomDistances()
        self.distance_expansion = distance_expansion

        # interaction blocks and post-interaction layers
        self.n_interactions = len(interactions)

        if pre_interactions is None:
            pre_interactions = nn.ModuleList(
                nn.Identity() for _ in range(len(interactions))
            )
        self.pre_interactions = pre_interactions

        self.interactions = interactions

        if interaction_refinements is None:
            interaction_refinements = nn.ModuleList(
                spk.nn.FeatureSum([]) for _ in range(len(interactions))
            )
        self.interaction_refinements = interaction_refinements

        if post_interactions is None:
            post_interactions = nn.ModuleList(
                nn.Identity() for _ in range(len(interactions))
            )
        self.post_interactions = post_interactions

        if interaction_outputs is None:
            interaction_outputs = nn.ModuleList(
                nn.Identity() for _ in range(len(interactions))
            )
        self.interaction_outputs = interaction_outputs

        # aggregation layer intermediate interactions
        self.interaction_aggregation = interaction_aggregation

    @property
    def n_atom_basis(self):
        return self.interactions[0].n_atom_basis

    def forward(self, inputs):
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        if Properties.charges in inputs.keys():
            charges = inputs[Properties.charges]
        else:
            charges = None
        if Properties.spin in inputs.keys():
            spins = inputs[Properties.spin]
        else:
            spins = None

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distance_provider(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)

        # compute intermediate interactions
        intermediate_interactions = []
        for (
            pre_interaction,
            interaction,
            post_interaction,
            interaction_refinement,
            interaction_output,
        ) in zip(
            self.pre_interactions,
            self.interactions,
            self.post_interactions,
            self.interaction_refinements,
            self.interaction_outputs,
        ):
            # pre interaction
            x = pre_interaction(x)

            # interaction layer x+v
            v = interaction(
                x,
                r_ij,
                neighbors,
                neighbor_mask,
                f_ij=f_ij,
                charges=charges,
                spins=spins,
                atom_mask=atom_mask,
            )

            # residual sum
            x = x + v

            # interaction refinement
            refinement_features = interaction_refinement({**inputs, "x": x})
            x = x + refinement_features

            # post interaction layer
            x = post_interaction(x)

            # output layer for interaction blocks
            y = interaction_output(x)

            # collect intermediate results
            intermediate_interactions.append(y)

        # get representation by aggregating intermediate interactions or selecting
        # last interaction output
        representation = self.interaction_aggregation(intermediate_interactions, x)

        # build results dict
        results = dict(representation=representation)
        if self.return_intermediate:
            results.update(dict(intermediate_interactions=intermediate_interactions))
        if self.return_distances:
            results.update(dict(distances=r_ij))

        return results
