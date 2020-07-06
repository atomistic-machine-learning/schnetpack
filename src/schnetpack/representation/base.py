import torch
import torch.nn as nn
import schnetpack as spk
from schnetpack import Properties


__all__ = ["InteractionAggregation", "AtomisticRepresentation"]


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


class AtomisticRepresentation(nn.Module):
    def __init__(
        self,
        embedding,
        distance_expansion,
        interactions,
        post_interactions=None,
        interaction_aggregation=InteractionAggregation(mode="last"),
        return_intermediate=False,
        return_distances=False,
        sum_before_interaction_append=False,
    ):
        # initialize layer
        super(AtomisticRepresentation, self).__init__()

        # attributes
        self.return_intermediate = return_intermediate
        self.return_distances = return_distances
        self.sum_before_interaction_append = sum_before_interaction_append

        # modules
        self.embedding = embedding
        self.distance_provider = spk.nn.AtomDistances()
        self.distance_expansion = distance_expansion
        self.interactions = interactions
        if post_interactions is None:
            post_interactions = nn.ModuleList(
                nn.Identity() for _ in range(len(interactions))
            )
        self.post_interactions = post_interactions
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
        if Properties.spin in inputs[Properties.spin]:
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
        for interaction, post_interaction in zip(
            self.interactions, self.post_interactions
        ):
            v = interaction(
                x,
                r_ij,
                neighbors,
                neighbor_mask,
                f_ij=f_ij,
                charges=charges,
                spins=spins,
            )
            x = x + v
            if self.sum_before_interaction_append:
                intermediate_interactions.append(post_interaction(x))
            else:
                intermediate_interactions.append(post_interaction(v))

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
