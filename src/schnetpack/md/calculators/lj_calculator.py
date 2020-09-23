import torch
import torch.nn as nn
from torch.autograd import grad

import schnetpack as spk
import schnetpack.nn as spknn
from schnetpack import Properties
from schnetpack.md.calculators import SchnetPackCalculator


class PairwiseRepresentation(nn.Module):
    def __init__(self):
        super(PairwiseRepresentation, self).__init__()
        self.distances = spknn.AtomDistances()

    def forward(self, inputs):
        positions = inputs[Properties.R]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        cell = inputs[Properties.cell]
        cell_offsets = inputs[Properties.cell_offset]

        distances = self.distances(
            positions,
            neighbors,
            cell=cell,
            cell_offsets=cell_offsets,
            neighbor_mask=neighbor_mask,
        )

        return distances


class LJAtomistic(nn.Module):
    def __init__(
        self,
        r_equilibrium=1.0,
        well_depth=10.0,
        cutoff=5.0,
        property="y",
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=True,
    ):
        super(LJAtomistic, self).__init__()

        self.r_equilibrium = r_equilibrium
        self.well_depth = well_depth

        self.create_graph = create_graph
        self.property = property
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        self.cutoff = CustomCutoff(cutoff)
        self.atom_pool = spknn.base.Aggregate(axis=1, mean=False)

    def forward(self, inputs):
        atom_mask = inputs[Properties.atom_mask]
        neighbor_mask = inputs[Properties.neighbor_mask]
        distances = inputs["representation"]

        # Compute lennard jones potential
        power_6 = torch.where(
            neighbor_mask == 1,
            (self.r_equilibrium / distances) ** 6,
            torch.zeros_like(distances),
        )
        r_cut = self.cutoff(distances) * neighbor_mask

        yi = 0.5 * torch.sum((power_6 ** 2 - power_6) * r_cut, dim=2)[:, :, None]

        y = self.well_depth * self.atom_pool(yi, atom_mask)

        # collect results
        result = {self.property: y}

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0] * torch.cross(cell[:, 1], cell[:, 2]), dim=1, keepdim=True
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

        return result


class CustomCutoff(nn.Module):
    def __init__(self, cutoff_radius, healing_length=0.3405):
        super(CustomCutoff, self).__init__()
        self.register_buffer("cutoff_radius", torch.Tensor([cutoff_radius]))
        self.register_buffer("healing_length", torch.Tensor([healing_length]))

    def forward(self, distances):
        r = (
            distances - (self.cutoff_radius - self.healing_length)
        ) / self.healing_length
        r_function = 1.0 + r ** 2 * (2.0 * r - 3.0)

        switch = torch.where(
            distances > self.cutoff_radius - self.healing_length,
            r_function,
            torch.ones_like(distances),
        )
        switch = torch.where(
            distances > self.cutoff_radius, torch.zeros_like(distances), switch
        )

        return switch


class LJCalculator(SchnetPackCalculator):
    def __init__(
        self,
        r_equilibrium,
        well_depth,
        required_properties,
        force_handle,
        position_conversion="Angstrom",
        force_conversion="eV / Angstrom",
        property_conversion={},
        stress_handle=None,
        stress_conversion="eV / Angstrom / Angstrom / Angstrom",
        detach=True,
        neighbor_list=spk.md.SimpleNeighborList,
        cutoff=-1.0,
        cutoff_shell=1.0,
        cutoff_lr=None,
        device="cpu",
    ):
        model = spk.atomistic.AtomisticModel(
            PairwiseRepresentation(),
            LJAtomistic(
                r_equilibrium=r_equilibrium,
                well_depth=well_depth,
                cutoff=cutoff,
                property=Properties.energy,
                negative_dr=True,
                derivative=force_handle,
                stress=stress_handle,
            ),
        ).to(device)

        super(LJCalculator, self).__init__(
            model,
            required_properties,
            force_handle,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
            stress_handle=stress_handle,
            stress_conversion=stress_conversion,
            detach=detach,
            neighbor_list=neighbor_list,
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
            cutoff_lr=cutoff_lr,
        )
