import torch

from schnetpack.md.simulation_hooks import SimulationHook

__all__ = [
    'BiasPotential',
    'AcceleratedMD',
    'MetaDyn',
    'CollectiveVariable',
    'BondColvar'
]


class BiasPotential(SimulationHook):
    """
    Placeholder class for bias potentials used for adaptive/accelerated sampling.
    """

    def __init__(self):
        raise NotImplementedError

    def on_step_end(self, simulator):
        raise NotImplementedError


class AcceleratedMD(SimulationHook):

    def __init__(self, energy_threshold,
                 acceleration_factor,
                 energy_handle='energy',
                 energy_conversion=1.0):
        self.energy_threshold = energy_threshold
        self.acceleration_factor = acceleration_factor
        self.energy_handle = energy_handle
        self.energy_conversion = energy_conversion

    def on_step_middle(self, simulator):
        energies, forces = self._get_energies_forces(simulator)

        # Compute bias potential and derivatives
        bias_potential = self._function(energies)
        print(energies, bias_potential, energies + bias_potential)
        bias_forces = self._derivative(energies) * forces

        # Store bias potential and update forces
        simulator.system.properties['bias_potential'] = bias_potential
        simulator.system.forces = forces + bias_forces.detach()

    def _get_energies_forces(self, simulator):
        energies = simulator.system.properties[self.energy_handle].float() * self.energy_conversion
        forces = simulator.system.forces
        return energies, forces

    def _function(self, energies):
        function = torch.pow(self.energy_threshold - energies, 2.0) / (
                self.acceleration_factor + self.energy_threshold - energies)
        function[energies >= self.energy_threshold] = 0
        return function

    def _derivative(self, energies):
        derivative = self.acceleration_factor ** 2 / torch.pow(
            self.acceleration_factor + self.energy_threshold - energies, 2.0) - 1.0
        derivative[energies >= self.energy_threshold] = 0
        return derivative


class CollectiveVariable:

    def __init__(self, width):
        # Initialize the width of the Gaussian
        self.width = 0.5 * width ** 2

    def get_colvar(self, structure):
        return self._colvar_function(structure)

    def _colvar_function(self, structure):
        raise NotImplementedError


class BondColvar(CollectiveVariable):

    def __init__(self, idx_a, idx_b, width):
        super(BondColvar, self).__init__(width)
        self.idx_a = idx_a
        self.idx_b = idx_b

    def _colvar_function(self, structure):
        vector_ab = structure[:, :, self.idx_b, :] - structure[:, :, self.idx_a, :]
        return torch.norm(vector_ab, 2, dim=2)


class MetaDyn(SimulationHook):

    def __init__(self,
                 collective_variables,
                 frequency=200,
                 weight=1.0 / 627.509
                 ):
        self.collective_variables = collective_variables

        self.gaussian_centers = None
        self.gaussian_mask = None
        self.collective_variable_widths = None

        self.frequency = frequency
        self.weigth = weight
        self.n_gaussians = 0

    def on_simulation_start(self, simulator):
        n_gaussian_centers = int(simulator.n_steps / self.frequency) + 1
        self.gaussian_centers = torch.zeros(n_gaussian_centers, len(self.collective_variables),
                                            device=simulator.system.device)
        self.collective_variable_widths = torch.FloatTensor([cv.width for cv in self.collective_variables],
                                                            device=simulator.system.device)
        self.gaussian_mask = torch.zeros(n_gaussian_centers, device=simulator.system.device)
        self.gaussian_mask[0] = 1

    def on_step_middle(self, simulator):
        # Get and detach the structure from the simulator
        structure = simulator.system.positions.detach()
        # Enable gradients for bias forces
        structure.requires_grad = True

        # Compute the collective variables
        colvars = torch.stack([colvar.get_colvar(structure) for colvar in self.collective_variables], dim=2)

        # Compute the Gaussians for the potential
        gaussians = torch.exp(
            -(colvars[:, :, None, :] - self.gaussian_centers[None, None, :, :]) ** 2
            / self.collective_variable_widths[None, None, None, :]
        )
        # Compute the bias potential and apply mask for centers not yet stored
        bias_potential = torch.prod(gaussians, dim=3) * self.gaussian_mask

        # Finalize potential and compute forces
        bias_potential = torch.sum(self.weigth * bias_potential, 2)
        bias_forces = -torch.autograd.grad(bias_potential, structure, torch.ones_like(bias_potential))[0]
        # print(bias_forces[0, 0])
        # print(bias_potential)

        # Store bias potential, collective variables and update system forces
        simulator.system.properties['bias_potential'] = bias_potential.detach()
        simulator.system.properties['collective_variables'] = colvars.detach()
        simulator.system.forces = simulator.system.forces + bias_forces.detach()

        # Add a new Gaussian to the potential every n_steps
        if simulator.step % self.frequency == 0:
            self.gaussian_centers[self.n_gaussians] = colvars.detach()
            # Update the mask
            self.gaussian_mask[self.n_gaussians + 1] = 1
            self.n_gaussians += 1
