import torch
import copy
import os
import logging

from schnetpack.md.simulation_hooks import SimulationHook
from schnetpack.md.initial_conditions import MaxwellBoltzmannInit
from schnetpack.data import AtomsData


class AdaptiveSamplingHook(SimulationHook):
    def __init__(
        self,
        thresholds,
        n_samples,
        dataset,
        reset=True,
        temperature=300,
        initializer=MaxwellBoltzmannInit,
    ):

        self.thresholds = thresholds
        self.n_samples = n_samples
        self.samples = []
        self.samples_thresholds = []

        # Reinitialization
        self.reset = reset
        self.initializer = initializer(temperature)

        # Dataset for storage
        self.dataset = dataset
        if os.path.exists(self.dataset):
            logging.info(
                "Database {:s} already exists. Data will be appended.".format(
                    self.dataset
                )
            )

        # Initial system variables for reset
        self.init_positions = None
        self.init_forces = None
        self.init_cells = None

    def on_simulation_start(self, simulator):
        # Store initial configs for system reset
        self.init_positions = copy.deepcopy(simulator.system.positions)
        self.init_forces = copy.deepcopy(simulator.system.forces)
        self.init_cells = copy.deepcopy(simulator.system.cells)

    def on_step_end(self, simulator):
        # Check if a sample is required
        sample_system, sample_molecule, threshold_exceeded = self._check_uncertainty(
            simulator.system
        )

        if sample_system:
            # Collect samples based on uncertainty thresholds
            self._collect_samples(simulator.system, sample_molecule, threshold_exceeded)

            # Reinitialize velocities if requested
            if self.reset:
                logging.info("Resetting system...")
                self._reset_system(simulator.system, mask=sample_molecule)

        # If a sufficient number of samples is collected, stop sampling
        if len(self.samples) >= self.n_samples:
            self._write_database()
            exit()

    def on_simulation_end(self, simulator):
        # Wrap everything up if simulation finished without collecting all/any samples.
        self._write_database()

    def _check_uncertainty(self, system):
        threshold_exceeded = {}

        sample_system = False
        sample_molecule = torch.zeros(
            system.n_replicas, system.n_molecules, device=system.device
        ).bool()

        # Check if a sample is needed
        for prop in self.thresholds:
            # Get variance from simulator
            prop_var = system.properties["{:s}_var".format(prop)]

            # Reshaping depending if property is atomic or not
            shapes = prop_var.shape
            if shapes[2] == system.max_n_atoms:
                prop_var.view(*shapes[:2], -1)
                uncertainty = torch.sqrt(torch.sum(prop_var, dim=-1))
            else:
                prop_var.view(*shapes[:1], -1)
                uncertainty = torch.sqrt(torch.sum(prop_var, dim=-1, keepdim=True))

            # Check if uncertainty threshold is exceeded
            threshold_exceeded[prop] = self.thresholds[prop] < uncertainty

            # Checks if a) a sample is needed, b) for which replica/molecule a sample is needed, c) for which atom
            if torch.any(threshold_exceeded[prop]):
                # Check for which molecule/replica samples are required
                sample_molecule = sample_molecule | torch.any(
                    threshold_exceeded[prop], dim=-1
                )
                # Overall sample required
                sample_system = True

        return sample_system, sample_molecule, threshold_exceeded

    def _collect_samples(self, system, sample_molecule, threshold_exceeded):
        # Get structures in the form of ASE atoms (R x M is flattened)
        atoms = system.get_ase_atoms(internal_units=False)

        # Collect all replicas and molecules which need sampling
        idx_c = 0
        for idx_r in range(system.n_replicas):
            for idx_m in range(system.n_molecules):

                # Get the atoms and store the thresholds
                if sample_molecule[idx_r, idx_m] == 1:
                    self.samples.append(atoms[idx_c])
                    sample_thresholds = {}
                    for prop in threshold_exceeded:
                        sample_thresholds[prop] = (
                            threshold_exceeded[prop][idx_r, idx_m]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    self.samples_thresholds.append(sample_thresholds)
                idx_c += 1

    def _reset_system(self, system, mask=None):
        if mask is None:
            system.positions = copy.deepcopy(self.init_positions)
            system.forces = copy.deepcopy(self.init_forces)
            system.cells = copy.deepcopy(self.init_cells)
            self.initializer.initialize_system(system)
        else:
            system.positions[mask == 1, ...] = copy.deepcopy(
                self.init_positions[mask == 1, ...]
            )
            system.forces[mask == 1, ...] = copy.deepcopy(
                self.init_forces[mask == 1, ...]
            )
            if self.init_cells is not None:
                system.cells[mask == 1, ...] = copy.deepcopy(
                    self.init_cells[mask == 1, ...]
                )
            self.initializer.initialize_system(system, mask=mask)

    def _write_database(self):
        if len(self.samples) > 0:
            dataset = AtomsData(
                self.dataset, available_properties=self.samples_thresholds[0]
            )
            dataset.add_systems(self.samples, self.samples_thresholds)
            logging.info(
                "{:d} samples written to {:s}.".format(len(self.samples), self.dataset)
            )
        else:
            logging.info("No samples collected.")
