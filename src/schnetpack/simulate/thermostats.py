import torch
import numpy as np
import scipy.linalg as linalg
import logging

from schnetpack.md.utils import MDUnits, load_gle_matrices, \
    NormalModeTransformer, YSWeights
from schnetpack.md.integrators import RingPolymer
from schnetpack.simulate.hooks import SimulationHook


class ThermostatError(Exception):
    pass


class ThermostatHook(SimulationHook):
    # TODO: Could be made a torch nn.Module

    def __init__(self, temperature_bath, nm_transformation=None, detach=True):
        self.temperature_bath = temperature_bath
        self.initialized = False
        self.device = None
        self.n_replicas = None
        self.nm_transformation = nm_transformation
        self.detach = detach

    def on_simulation_start(self, simulator):
        self.device = simulator.system.device
        self.n_replicas = simulator.system.n_replicas

        # Check if using normal modes is feasible and initialize
        if self.nm_transformation is not None:
            if type(simulator.integrator) is not RingPolymer:
                raise ThermostatError('Normal mode transformation should only be used with ring polymer dynamics.')
            else:
                self.nm_transformation = self.nm_transformation(self.n_replicas, device=self.device)

        if not self.initialized:
            self._init_thermostat(simulator)
            self.initialized = True

    def on_step_begin(self, simulator):
        # Apply thermostat
        self._apply_thermostat(simulator)

        # Re-apply atom masks for differently sized molecules, as some thermostats add random noise
        simulator.system.momenta = simulator.system.momenta * simulator.system.atom_masks

        # Detach if requested
        if self.detach:
            simulator.system.momenta = simulator.system.momenta.detach()

    def on_step_end(self, simulator):
        # Apply thermostat
        self._apply_thermostat(simulator)

        # Re-apply atom masks for differently sized molecules, as some thermostats add random noise
        simulator.system.momenta = simulator.system.momenta * simulator.system.atom_masks

        # Detach if requested
        if self.detach:
            simulator.system.momenta = simulator.system.momenta.detach()

    def _init_thermostat(self, simulator):
        pass

    def _apply_thermostat(self, simulator):
        raise NotImplementedError


class BerendsenThermostat(ThermostatHook):

    def __init__(self, temperature_bath, time_constant):
        super(BerendsenThermostat, self).__init__(temperature_bath)

        self.time_constant = time_constant * MDUnits.fs2atu

    def _apply_thermostat(self, simulator):
        scaling = 1.0 + simulator.integrator.time_step / self.time_constant * (
                self.temperature_bath / simulator.system.temperature - 1)
        simulator.system.momenta = torch.sqrt(scaling[:, :, None, None]) * simulator.system.momenta


class GLEThermostat(ThermostatHook):

    def __init__(self, bath_temperature, gle_file, nm_transformation=None):
        super(GLEThermostat, self).__init__(bath_temperature,
                                            nm_transformation=nm_transformation)

        self.gle_file = gle_file

        # To be initialized on beginning of the simulation, once system and integrator are known
        self.c1 = None
        self.c2 = None
        self.thermostat_momenta = None
        self.thermostat_factor = None

    def _init_thermostat(self, simulator):
        # Generate main matrices
        self.c1, self.c2 = self._init_gle_matrices(simulator)

        # Get particle masses
        self.thermostat_factor = torch.sqrt(simulator.system.masses)[..., None]

        # Get initial thermostat momenta
        self.thermostat_momenta = self._init_thermostat_momenta(simulator)

    def _init_gle_matrices(self, simulator):
        a_matrix, c_matrix = load_gle_matrices(self.gle_file)

        if a_matrix is None:
            raise ThermostatError('Error reading GLE matrices from {:s}'.format(self.gle_file))
        elif a_matrix.shape[0] > 1:
            raise ThermostatError('More than one A matrix found. Could be PIGLET input.')
        else:
            # Remove leading dimension (for normal modes)
            a_matrix = a_matrix.squeeze()

        c1, c2 = self._init_single_gle_matrix(a_matrix, c_matrix, simulator)
        return c1, c2

    def _init_single_gle_matrix(self, a_matrix, c_matrix, simulator):

        if c_matrix is None:
            c_matrix = np.eye(a_matrix.shape[-1]) * self.temperature_bath * MDUnits.kB
            # Check if normal GLE or GLE for ring polymers is needed:
            if type(simulator.integrator) is RingPolymer:
                logging.info('RingPolymer integrator detected, initializing C accordingly.')
                c_matrix *= simulator.system.n_replicas
        else:
            c_matrix = c_matrix.squeeze()
            logging.info('C matrix for GLE loaded, provided temperature will be ignored.')

        # A does not need to be transposed, else c2 is imaginary
        c1 = linalg.expm(-0.5 * simulator.integrator.time_step * a_matrix)

        # c2 is symmetric
        c2 = linalg.sqrtm(c_matrix - np.dot(c1, np.dot(c_matrix, c1.T)))

        c1 = torch.from_numpy(c1.T).to(self.device).float()
        c2 = torch.from_numpy(c2).to(self.device).float()
        return c1, c2

    def _init_thermostat_momenta(self, simulator, free_particle_limit=True):
        degrees_of_freedom = self.c1.shape[-1]
        if not free_particle_limit:
            initial_momenta = torch.zeros(*simulator.system.momenta.shape, degrees_of_freedom, device=self.device)
        else:
            initial_momenta = torch.randn(*simulator.system.momenta.shape, degrees_of_freedom, device=self.device)
            initial_momenta = torch.matmul(initial_momenta, self.c2)
        return initial_momenta

    def _apply_thermostat(self, simulator):
        # Generate random noise
        thermostat_noise = torch.randn(self.thermostat_momenta.shape, device=self.device)

        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        # Set current momenta
        self.thermostat_momenta[:, :, :, :, 0] = momenta

        # Apply thermostat
        self.thermostat_momenta = torch.matmul(self.thermostat_momenta, self.c1) + \
                                  torch.matmul(thermostat_noise, self.c2) * self.thermostat_factor

        # Extract momenta
        momenta = self.thermostat_momenta[:, :, :, :, 0]

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta

    @property
    def state_dict(self):
        state_dict = {
            'c1': self.c1,
            'c2': self.c2,
            'thermostat_factor': self.thermostat_factor,
            'thermostat_momenta': self.thermostat_momenta,
            'temperature_bath': self.temperature_bath,
            'n_replicas': self.n_replicas
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict['c1']
        self.c2 = state_dict['c2']
        self.thermostat_factor = state_dict['thermostat_factor']
        self.thermostat_momenta = state_dict['thermostat_momenta']
        self.temperature_bath = state_dict['temperature_bath']
        self.n_replicas = state_dict['n_replicas']

        # Set initialized flag
        self.initialized = True


class PIGLETThermostat(GLEThermostat):

    def __init__(self, temperature_bath, gle_file,
                 nm_transformation=NormalModeTransformer):
        super(PIGLETThermostat, self).__init__(temperature_bath, gle_file, nm_transformation=nm_transformation)

    def _init_gle_matrices(self, simulator):
        a_matrix, c_matrix = load_gle_matrices(self.gle_file)

        if a_matrix is None:
            raise ThermostatError('Error reading GLE matrices from {:s}'.format(self.gle_file))
        if a_matrix.shape[0] != self.n_replicas:
            raise ThermostatError('Expected {:d} beads but found {:d}.'.format(a_matrix.shape[0], self.n_replicas))
        if not type(simulator.integrator) is RingPolymer:
            raise ThermostatError('PIGLET thermostat should only be used with RPMD.')

        all_c1 = []
        all_c2 = []

        # Generate main matrices
        for b in range(self.n_replicas):
            c1, c2 = self._init_single_gle_matrix(a_matrix[b], (c_matrix[b], None)[c_matrix is None], simulator)
            # Add extra dimension for use with torch.cat, correspond to normal modes of ring polymer
            all_c1.append(c1[None, ...])
            all_c2.append(c2[None, ...])

        # Bring to correct shape for later matmul broadcasting
        c1 = torch.cat(all_c1)[:, None, None, :, :]
        c2 = torch.cat(all_c2)[:, None, None, :, :]
        return c1, c2


class LangevinThermostat(ThermostatHook):

    def __init__(self, temperature_bath, time_constant, nm_transformation=None):
        super(LangevinThermostat, self).__init__(temperature_bath, nm_transformation=nm_transformation)

        self.time_constant = time_constant * MDUnits.fs2atu

        self.thermostat_factor = None
        self.c1 = None
        self.c2 = None

    def _init_thermostat(self, simulator):
        # Initialize friction coefficients
        gamma = torch.ones(1, device=self.device) / self.time_constant

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma)
        c2 = torch.sqrt(1 - c1 ** 2)

        self.c1 = c1.to(self.device)[:, None, None, None]
        self.c2 = c2.to(self.device)[:, None, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(simulator.system.masses * MDUnits.kB * self.temperature_bath)

    def _apply_thermostat(self, simulator):
        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        # Generate random noise
        thermostat_noise = torch.randn(momenta.shape, device=self.device)

        # Apply thermostat
        momenta = self.c1 * momenta + self.thermostat_factor * self.c2 * thermostat_noise

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta

    @property
    def state_dict(self):
        state_dict = {
            'c1': self.c1,
            'c2': self.c2,
            'thermostat_factor': self.thermostat_factor,
            'temperature_bath': self.temperature_bath,
            'n_replicas': self.n_replicas
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict['c1']
        self.c2 = state_dict['c2']
        self.thermostat_factor = state_dict['thermostat_factor']
        self.temperature_bath = state_dict['temperature_bath']
        self.n_replicas = state_dict['n_replicas']

        # Set initialized flag
        self.initialized = True


class PILELocalThermostat(LangevinThermostat):

    def __init__(self, temperature_bath, time_constant, nm_transformation=NormalModeTransformer):
        super(PILELocalThermostat, self).__init__(temperature_bath, time_constant, nm_transformation=nm_transformation)

    def _init_thermostat(self, simulator):
        if type(simulator.integrator) is not RingPolymer:
            raise ThermostatError('PILE thermostats can only be used in RPMD')

        # Initialize friction coefficients
        gamma_normal = 2 * simulator.integrator.omega_normal
        # Use seperate coefficient for centroid mode
        gamma_normal[0] = 1.0 / self.time_constant

        if self.nm_transformation is None:
            raise ThermostatError('Normal mode transformation required for PILE thermostat')

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma_normal)
        c2 = torch.sqrt(1 - c1 ** 2)

        self.c1 = c1.to(self.device)[:, None, None, None]
        self.c2 = c2.to(self.device)[:, None, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(
            simulator.system.masses * MDUnits.kB * self.n_replicas * self.temperature_bath
        )

    @property
    def state_dict(self):
        state_dict = {
            'c1': self.c1,
            'c2': self.c2,
            'thermostat_factor': self.thermostat_factor,
            'temperature_bath': self.temperature_bath,
            'n_replicas': self.n_replicas
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict['c1']
        self.c2 = state_dict['c2']
        self.thermostat_factor = state_dict['thermostat_factor']
        self.temperature_bath = state_dict['temperature_bath']
        self.n_replicas = state_dict['n_replicas']

        # Set initialized flag
        self.initialized = True


class PILEGlobalThermostat(PILELocalThermostat):

    def __init__(self, temperature_bath, time_constant, nm_transformation=NormalModeTransformer):
        super(PILEGlobalThermostat, self).__init__(temperature_bath, time_constant,
                                                   nm_transformation=nm_transformation)

    def _apply_thermostat(self, simulator):
        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        # Generate random noise
        thermostat_noise = torch.randn(momenta.shape, device=self.device)

        # Apply thermostat to centroid mode
        c1_centroid = self.c1[0]
        momenta_centroid = momenta[0]
        thermostat_noise_centroid = thermostat_noise[0]

        # Compute kinetic energy of centroid
        kinetic_energy_factor = torch.sum(momenta_centroid ** 2 / simulator.system.masses[0]) / (
                self.temperature_bath * MDUnits.kB * self.n_replicas)

        centroid_factor = (1 - c1_centroid) / kinetic_energy_factor

        alpha_sq = c1_centroid + torch.sum(thermostat_noise_centroid ** 2) * centroid_factor + \
                   2 * thermostat_noise_centroid[0, 0, 0] * torch.sqrt(c1_centroid * centroid_factor)

        alpha_sign = torch.sign(thermostat_noise_centroid[0, 0, 0] + torch.sqrt(c1_centroid / centroid_factor))

        alpha = torch.sqrt(alpha_sq) * alpha_sign

        # Finally apply thermostat...
        momenta[0] = alpha * momenta[0]

        # Apply thermostat for remaining normal modes
        momenta[1:] = self.c1[1:] * momenta[1:] + self.thermostat_factor * self.c2[1:] * thermostat_noise[1:]

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta


class NHCThermostat(ThermostatHook):

    def __init__(self, temperature_bath, time_constant, chain_length=3, massive=False,
                 nm_transformation=None, multi_step=2, integration_order=3):
        super(NHCThermostat, self).__init__(temperature_bath, nm_transformation=nm_transformation)

        self.chain_length = chain_length
        self.massive = massive
        self.frequency = 1 / (time_constant * MDUnits.fs2atu)

        # Cpmpute kBT, since it will be used a lot
        self.kb_temperature = self.temperature_bath * MDUnits.kB

        # Propagation parameters
        self.multi_step = multi_step
        self.integration_order = integration_order
        self.time_step = None

        # Find out number of particles (depends on whether massive or not)
        self.degrees_of_freedom = None
        self.masses = None

        self.velocities = None
        self.positions = None
        self.forces = None

    def _init_thermostat(self, simulator):
        # Determine integration step via multi step and Yoshida Suzuki weights
        integration_weights = YSWeights(self.device).get_weights(self.integration_order)
        self.time_step = simulator.integrator.time_step * integration_weights / self.multi_step

        # Determine shape of tensors and internal degrees of freedom
        n_replicas, n_molecules, n_atoms, xyz = simulator.system.momenta.shape

        if self.massive:
            state_dimension = (n_replicas, n_molecules, n_atoms, xyz, self.chain_length)
            # Since momenta will be masked later, no need to set non-atoms to 0
            self.degrees_of_freedom = torch.ones((n_replicas, n_molecules, n_atoms, xyz), device=self.device)
        else:
            state_dimension = (n_replicas, n_molecules, 1, 1, self.chain_length)
            self.degrees_of_freedom = 3 * simulator.system.n_atoms.float()[None, :, None, None]

        # Set up masses
        self._init_masses(state_dimension, simulator)

        # Set up internal variables
        self.positions = torch.zeros(state_dimension, device=self.device)
        self.forces = torch.zeros(state_dimension, device=self.device)
        self.velocities = torch.zeros(state_dimension, device=self.device)

    def _init_masses(self, state_dimension, simulator):
        self.masses = torch.ones(state_dimension, device=self.device)
        # Get masses of innermost thermostat
        self.masses[..., 0] = self.degrees_of_freedom * self.kb_temperature / self.frequency ** 2
        # Set masses of remaining thermostats
        self.masses[..., 1:] = self.kb_temperature / self.frequency ** 2

    def _propagate_thermostat(self, kinetic_energy):
        # Compute forces on first thermostat
        self.forces[..., 0] = (kinetic_energy - self.degrees_of_freedom * self.kb_temperature) / self.masses[..., 0]

        scaling_factor = 1.0
        for _ in range(self.multi_step):
            for idx_ys in range(self.integration_order):
                time_step = self.time_step[idx_ys]

                # Update velocities of outermost bath
                self.velocities[..., -1] += 0.25 * self.forces[..., -1] * time_step

                # Update the velocities moving through the beads of the chain
                for chain in range(self.chain_length - 2, -1, -1):
                    coeff = torch.exp(-0.125 * time_step * self.velocities[..., chain + 1])
                    self.velocities[..., chain] = self.velocities[..., chain] * coeff ** 2 + \
                                                  0.25 * self.forces[..., chain] * coeff * time_step

                # Accumulate velocity scaling
                scaling_factor *= torch.exp(-0.5 * time_step * self.velocities[..., 0])
                # Update forces of innermost thermostat
                self.forces[..., 0] = (scaling_factor * scaling_factor * kinetic_energy
                                       - self.degrees_of_freedom * self.kb_temperature) / self.masses[..., 0]

                # Update thermostat positions
                # TODO: Only required if one is interested in the conserved quanity of the NHC.
                self.positions += 0.5 * self.velocities * time_step

                # Update the thermostat velocities
                for chain in range(self.chain_length - 1):
                    coeff = torch.exp(-0.125 * time_step * self.velocities[..., chain + 1])
                    self.velocities[..., chain] = self.velocities[..., chain] * coeff ** 2 + \
                                                  0.25 * self.forces[..., chain] * coeff * time_step
                    self.forces[..., chain + 1] = (self.masses[..., chain] * self.velocities[..., chain] ** 2
                                                   - self.kb_temperature) / self.masses[..., chain + 1]

                # Update velocities of outermost thermostat
                self.velocities[..., -1] += 0.25 * self.forces[..., -1] * time_step

        return scaling_factor

    def _compute_kinetic_energy(self, momenta, masses):
        # Compute the kinetic energy (factor of 1/2 can be removed, as it cancels with a times 2)
        # TODO: Is no problem, as NM transformation never mixes atom dimension which carries the masses.
        kinetic_energy = momenta ** 2 / masses
        if self.massive:
            return kinetic_energy
        else:
            return torch.sum(torch.sum(kinetic_energy, 3, keepdim=True), 2, keepdim=True)

    def _apply_thermostat(self, simulator):
        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        kinetic_energy = self._compute_kinetic_energy(momenta, simulator.system.masses)

        scaling_factor = self._propagate_thermostat(kinetic_energy)
        momenta = momenta * scaling_factor

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta

    @property
    def state_dict(self):
        state_dict = {
            'chain_length': self.chain_length,
            'massive': self.massive,
            'frequency': self.frequency,
            'kb_temperature': self.kb_temperature,
            'degrees_of_freedom': self.degrees_of_freedom,
            'masses': self.masses,
            'velocities': self.velocities,
            'forces': self.forces,
            'positions': self.positions,
            'time_step': self.time_step,
            'temperature_bath': self.temperature_bath,
            'n_replicas': self.n_replicas,
            'multi_step': self.multi_step,
            'integration_order': self.integration_order
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.chain_length = state_dict['chain_length']
        self.massive = state_dict['massive']
        self.frequency = state_dict['frequency']
        self.kb_temperature = state_dict['kb_temperature']
        self.degrees_of_freedom = state_dict['degrees_of_freedom']
        self.masses = state_dict['masses']
        self.velocities = state_dict['velocities']
        self.forces = state_dict['forces']
        self.positions = state_dict['positions']
        self.time_step = state_dict['time_step']
        self.temperature_bath = state_dict['temperature_bath']
        self.n_replicas = state_dict['n_replicas']
        self.multi_step = state_dict['multi_step']
        self.integration_order = state_dict['integration_order']

        self.initialized = True


class NHCRingPolymerThermostat(NHCThermostat):

    def __init__(self, temperature_bath, time_constant, chain_length=3, local=True,
                 nm_transformation=NormalModeTransformer, multi_step=2, integration_order=3):
        super(NHCRingPolymerThermostat, self).__init__(temperature_bath,
                                                       time_constant,
                                                       chain_length=chain_length,
                                                       massive=True,
                                                       nm_transformation=nm_transformation,
                                                       multi_step=multi_step,
                                                       integration_order=integration_order)
        self.local = local

    def _init_masses(self, state_dimension, simulator):
        # Multiply factor by number of replicas
        self.kb_temperature = self.kb_temperature * self.n_replicas

        # Initialize masses with the frequencies of the ring polymer
        polymer_frequencies = simulator.integrator.omega_normal
        polymer_frequencies[0] = 0.5 * self.frequency  # 0.5 comes from Ceriotti paper, check

        # Assume standard massive Nose-Hoover and initialize accordingly
        self.masses = torch.ones(state_dimension, device=self.device)
        self.masses *= self.kb_temperature / polymer_frequencies[:, None, None, None, None] ** 2

        # If a global thermostat is requested, we assign masses of 3N to the first link in the chain on the centroid
        if not self.local:
            self.masses[0, :, :, :, 0] *= 3 * simulator.system.n_atoms.float()[:, None, None]
            # Degrees of freedom also need to be adapted
            self.degrees_of_freedom[0, :, :, :] *= 3 * simulator.system.n_atoms.float()[:, None, None]

    def _compute_kinetic_energy(self, momenta, masses):
        kinetic_energy = momenta ** 2 / masses

        # In case of a global NHC for RPMD, use the whole centroid kinetic energy and broadcast it
        if not self.local:
            kinetic_energy_centroid = torch.sum(torch.sum(kinetic_energy[0, ...], 2, keepdim=True), 1, keepdim=True)
            kinetic_energy[0, ...] = kinetic_energy_centroid

        return kinetic_energy

    @property
    def state_dict(self):
        state_dict = {
            'chain_length': self.chain_length,
            'massive': self.massive,
            'frequency': self.frequency,
            'kb_temperature': self.kb_temperature,
            'degrees_of_freedom': self.degrees_of_freedom,
            'masses': self.masses,
            'velocities': self.velocities,
            'forces': self.forces,
            'positions': self.positions,
            'time_step': self.time_step,
            'temperature_bath': self.temperature_bath,
            'n_replicas': self.n_replicas,
            'multi_step': self.multi_step,
            'integration_order': self.integration_order,
            'local': self.local
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.chain_length = state_dict['chain_length']
        self.massive = state_dict['massive']
        self.frequency = state_dict['frequency']
        self.kb_temperature = state_dict['kb_temperature']
        self.degrees_of_freedom = state_dict['degrees_of_freedom']
        self.masses = state_dict['masses']
        self.velocities = state_dict['velocities']
        self.forces = state_dict['forces']
        self.positions = state_dict['positions']
        self.time_step = state_dict['time_step']
        self.temperature_bath = state_dict['temperature_bath']
        self.n_replicas = state_dict['n_replicas']
        self.multi_step = state_dict['multi_step']
        self.integration_order = state_dict['integration_order']
        self.local = state_dict['local']

        self.initialized = True
