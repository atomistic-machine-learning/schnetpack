from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from torch.autograd import grad

from schnetpack.nn.utils import derivative_from_molecular, derivative_from_atomic
import schnetpack.properties as properties

__all__ = ["Forces", "Strain", "Response"]


class ResponseException(Exception):
    pass


class Forces(nn.Module):
    """
    Predicts forces and stress as response of the energy prediction
    w.r.t. the atom positions and strain.

    """

    def __init__(
        self,
        calc_forces: bool = True,
        calc_stress: bool = False,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        stress_key: str = properties.stress,
    ):
        """
        Args:
            calc_forces: If True, calculate atomic forces.
            calc_stress: If True, calculate the stress tensor.
            energy_key: Key of the energy in results.
            force_key: Key of the forces in results.
            stress_key: Key of the stress in results.
        """
        super(Forces, self).__init__()
        self.calc_forces = calc_forces
        self.calc_stress = calc_stress
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.model_outputs = []
        if calc_forces:
            self.model_outputs.append(force_key)
        if calc_stress:
            self.model_outputs.append(stress_key)

        self.required_derivatives = []
        if self.calc_forces:
            self.required_derivatives.append(properties.R)
        if self.calc_stress:
            self.required_derivatives.append(properties.strain)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Epred = inputs[self.energy_key]

        go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
        grads = grad(
            [Epred],
            [inputs[prop] for prop in self.required_derivatives],
            grad_outputs=go,
            create_graph=self.training,
        )

        if self.calc_forces:
            dEdR = grads[0]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdR is None:
                dEdR = torch.zeros_like(inputs[properties.R])

            inputs[self.force_key] = -dEdR

        if self.calc_stress:
            stress = grads[-1]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if stress is None:
                stress = torch.zeros_like(inputs[properties.cell])

            cell = inputs[properties.cell]
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[:, :, None]
            inputs[self.stress_key] = stress / volume

        return inputs


class Response(nn.Module):
    implemented_properties = [
        properties.forces,
        properties.stress,
        properties.hessian,
        properties.dipole_moment,
        properties.polarizability,
        properties.dipole_derivatives,
        properties.partial_charges,
        properties.polarizability_derivatives,
        properties.shielding,
        properties.nuclear_spin_coupling,
    ]

    def __init__(
        self,
        energy_key: str,
        response_properties: List[str],
        map_properties: Optional[Dict[str, str]] = None,
    ):
        """
        Compute different response properties by taking derivatives of an energy model. See [#field1]_ for details.

        Args:
            energy_key (str): key indicating the energy property used for response calculations.
            response_properties (list(str)): List of requested response properties.
            map_properties (dict(str,str)):  Dictionary for mapping property names. The keys are the names as computed
                                             by the response layer (default `schnetpack.properties`), the values the
                                             new names.

        References:
        -----------
        .. [#field1] Gastegger, Schütt, Müller:
            Machine learning of solvent effects on molecular spectra and reactions.
            Chemical Science, 12(34), 11473-11483. 2021.
        """
        super(Response, self).__init__()

        for prop in response_properties:
            if prop not in self.implemented_properties:
                raise NotImplementedError(
                    "Property {:s} not implemented in response layer.".format(prop)
                )

        self.energy_key = energy_key
        self.response_properties = response_properties

        if map_properties is None:
            self.map_properties = {}
        else:
            self.map_properties = map_properties

        for prop in self.response_properties:
            if prop not in self.map_properties:
                self.map_properties[prop] = prop

        self.model_outputs = list(self.map_properties.keys())

        # Set up instructions for computing response properties and derivatives
        (
            basic_derivatives,
            required_derivatives,
            derivative_instructions,
            graph_required,
        ) = self._construct_properties()
        # Basic and required can not be merged
        self.basic_derivatives = basic_derivatives
        self.required_derivatives = required_derivatives
        self.derivative_instructions = derivative_instructions
        self.graph_required = graph_required

        # Check whether basic graph is enough or higher level derivatives are necessary
        self.basic_graph_required = len(self.basic_derivatives) != len(
            [p for p in self.derivative_instructions if self.derivative_instructions[p]]
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        energy = inputs[self.energy_key]

        # Compute base level derivatives
        go: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        basic_derivatives = grad(
            [energy],
            [inputs[prop] for prop in self.basic_derivatives.values()],
            grad_outputs=go,
            create_graph=(self.basic_graph_required or self.training),
            retain_graph=(self.basic_graph_required or self.training),
        )
        # Convert to dictionary
        basic_derivatives = dict(zip(self.basic_derivatives.keys(), basic_derivatives))
        results = {}

        # ================================
        # dE / dR
        # ================================
        if self.derivative_instructions["dEdR"]:

            # basic distance derivatives
            if properties.forces in self.response_properties:
                results[properties.forces] = -basic_derivatives["dEdR"]

            if self.derivative_instructions["d2EdR2"]:
                d2EdR2 = derivative_from_atomic(
                    basic_derivatives["dEdR"],
                    inputs[properties.R],
                    inputs[properties.n_atoms],
                    create_graph=(self.graph_required["d2EdR2"] or self.training),
                    retain_graph=True,
                )
                results[properties.hessian] = d2EdR2

        # ================================
        # dE / ds
        # ================================
        if self.derivative_instructions["dEds"]:
            stress = basic_derivatives["dEds"]

            # TorchScript needs Tensor instead of Optional[Tensor]
            if stress is None:
                stress = torch.zeros_like(inputs[properties.cell])

            cell = inputs[properties.cell]
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[:, :, None]
            results[properties.stress] = stress / volume

        # ================================
        # dE / dF
        # ================================
        if self.derivative_instructions["dEdF"]:
            dEdF = basic_derivatives["dEdF"]
            results[properties.dipole_moment] = -basic_derivatives["dEdF"]

            if self.derivative_instructions["d2EdFdR"]:
                d2EdFdR = derivative_from_molecular(
                    -dEdF,
                    inputs[properties.R],
                    create_graph=(self.graph_required["d2EdFdR"] or self.training),
                    retain_graph=True,
                )
                results[properties.dipole_derivatives] = d2EdFdR

                # Compute partial charges if requested
                if properties.partial_charges in self.response_properties:
                    results[properties.partial_charges] = (
                        torch.einsum("bii->b", d2EdFdR) / 3.0
                    )

            if self.derivative_instructions["d2EdF2"]:
                d2EdF2 = derivative_from_molecular(
                    -dEdF,
                    inputs[properties.electric_field],
                    create_graph=(self.graph_required["d2EdF2"] or self.training),
                    retain_graph=True,
                )
                results[properties.polarizability] = d2EdF2

                if self.derivative_instructions["d3EdF2dR"]:
                    d3EdF2dR = derivative_from_molecular(
                        d2EdF2,
                        inputs[properties.R],
                        create_graph=(self.graph_required["d3EdF2dR"] or self.training),
                        retain_graph=True,
                    )
                    results[properties.polarizability_derivatives] = d3EdF2dR

        # ================================
        # dE / dB
        # ================================
        if self.derivative_instructions["dEdB"]:
            dEdB = basic_derivatives["dEdB"]
            results["dEdB"] = dEdB

            if self.derivative_instructions["d2EdBdI"]:
                d2EdBdI = derivative_from_molecular(
                    dEdB,
                    inputs[properties.nuclear_magnetic_moments],
                    create_graph=(self.graph_required["d2EdBdI"] or self.training),
                    retain_graph=True,
                )
                results[properties.shielding] = d2EdBdI

        # ================================
        # dE / dI
        # ================================
        if self.derivative_instructions["dEdI"]:
            dEdI = basic_derivatives["dEdI"]
            results["dEdI"] = dEdI

            if self.derivative_instructions["d2EdI2"]:
                d2EdI2 = derivative_from_atomic(
                    dEdI,
                    inputs[properties.nuclear_magnetic_moments],
                    inputs[properties.n_atoms],
                    create_graph=(self.graph_required["d2EdI2"] or self.training),
                    retain_graph=True,
                )
                results[properties.nuclear_spin_coupling] = d2EdI2

        for prop in self.map_properties:
            inputs[self.map_properties[prop]] = results[prop]

        return inputs

    def _construct_properties(
        self,
    ) -> Tuple[Dict[str, str], List[str], Dict[str, bool], Dict[str, bool]]:
        """
        Routine for automatically determining the computational settings of the response
        layer based on the requested response properties.

        Based on the requested response properties, determine:
            - which derivatives need to be computed
            - which properties need to be enabled for gradient computation
            - for which derivatives does a graph need to be constructed

        Returns:
            - dictionary of basic derivatives
            - list of variables which need gradients
            - dictionary of derivative instructions
            - dictionary of required graphs
        """
        derivative_instructions = {
            "dEdR": False,
            "d2EdR2": False,
            "dEdF": False,
            "d2EdFdR": False,
            "d2EdF2": False,
            "d3EdF2dR": False,
            "dEdB": False,
            "dEdI": False,
            "d2EdBdI": False,
            "d2EdI2": False,
            "dEds": False,
        }
        graph_required = {
            "dEdR": False,
            "d2EdR2": False,
            "dEdF": False,
            "d2EdFdR": False,
            "d2EdF2": False,
            "d3EdF2dR": False,
            "dEdB": False,
            "dEdI": False,
            "d2EdBdI": False,
            "d2EdI2": False,
            "dEds": False,
        }

        required_derivatives = set()
        basic_derivatives = dict()

        # position derivatives
        if (properties.forces in self.response_properties) or (
            properties.hessian in self.response_properties
        ):

            derivative_instructions["dEdR"] = True
            required_derivatives.add(properties.R)
            basic_derivatives["dEdR"] = properties.R

            if properties.hessian in self.response_properties:
                graph_required["dEdR"] = True
                derivative_instructions["d2EdR2"] = True

        # strain derivatives
        if properties.stress in self.response_properties:
            derivative_instructions["dEds"] = True
            required_derivatives.add(properties.strain)
            basic_derivatives["dEds"] = properties.strain

        # electric field derivatives
        if (
            (properties.dipole_moment in self.response_properties)
            or (properties.polarizability in self.response_properties)
            or (properties.dipole_derivatives in self.response_properties)
            or (properties.polarizability_derivatives in self.response_properties)
            or (properties.partial_charges in self.response_properties)
        ):

            derivative_instructions["dEdF"] = True
            required_derivatives.add(properties.electric_field)
            basic_derivatives["dEdF"] = properties.electric_field

            if (properties.dipole_derivatives in self.response_properties) or (
                properties.partial_charges in self.response_properties
            ):
                graph_required["dEdF"] = True
                derivative_instructions["d2EdFdR"] = True
                required_derivatives.add(properties.R)

            if (properties.polarizability in self.response_properties) or (
                properties.polarizability_derivatives in self.response_properties
            ):
                graph_required["dEdF"] = True
                derivative_instructions["d2EdF2"] = True

                if properties.polarizability_derivatives in self.response_properties:
                    graph_required["d2EdF2"] = True
                    derivative_instructions["d3EdF2dR"] = True
                    required_derivatives.add(properties.R)

        # magnetic moment derivatives
        if properties.nuclear_spin_coupling in self.response_properties:
            # First derivative
            required_derivatives.add(properties.nuclear_magnetic_moments)
            basic_derivatives["dEdI"] = properties.nuclear_magnetic_moments
            derivative_instructions["dEdI"] = True

            # Second derivative for couplings
            graph_required["dEdI"] = True
            derivative_instructions["d2EdI2"] = True

        # magnetic field derivatives
        if properties.shielding in self.response_properties:
            # First derivative
            required_derivatives.add(properties.magnetic_field)
            basic_derivatives["dEdB"] = properties.magnetic_field
            derivative_instructions["dEdB"] = True

            # Second derivative for shielding
            required_derivatives.add(properties.nuclear_magnetic_moments)
            graph_required["dEdB"] = True
            derivative_instructions["d2EdBdI"] = True

        # Convert back to list
        required_derivatives = list(required_derivatives)

        return (
            basic_derivatives,
            required_derivatives,
            derivative_instructions,
            graph_required,
        )


class Strain(nn.Module):
    """
    This is required to calculate the stress as a response property.
    Adds strain-dependence to relative atomic positions Rij and (optionally) to absolute
    positions and unit cell.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]):
        strain = torch.zeros_like(inputs[properties.cell])
        strain.requires_grad_()
        inputs[properties.strain] = strain
        strain = strain.transpose(1, 2)

        # strain cell
        inputs[properties.cell] = inputs[properties.cell] + torch.matmul(
            inputs[properties.cell], strain
        )

        # strain positions
        idx_m = inputs[properties.idx_m]
        strain_i = strain[idx_m]
        inputs[properties.R] = inputs[properties.R] + torch.matmul(
            inputs[properties.R][:, None, :], strain_i
        ).squeeze(1)

        idx_i = inputs[properties.idx_i]
        strain_ij = strain_i[idx_i]
        inputs[properties.offsets] = inputs[properties.offsets] + torch.matmul(
            inputs[properties.offsets][:, None, :], strain_ij
        ).squeeze(1)
        return inputs
