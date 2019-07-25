"""
All logging operations in SchNetPack molecular dynamics simulations are performed via
simulation hooks. This includes the generation of checkpoint files. The main tool to
store simulation data is the :obj:`schnetpack.md.simulation_hooks.FileLogger`, which uses
data streams to collect information (positions, velocities, properties, ...) during a
simulation and store it to specially formatted HDF5 files. These files can then be read
using the :obj:`schnetpack.md.utils.HDF5Loader`.
"""

import json
import os
import h5py
import numpy as np
import torch

from schnetpack.md.simulation_hooks import SimulationHook

__all__ = [
    "Checkpoint",
    "TemperatureLogger",
    "FileLogger",
    "MoleculeStream",
    "DataStream",
    "PropertyStream",
    "SimulationStream",
]


class Checkpoint(SimulationHook):
    """
    Hook for writing out checkpoint files containing the state_dict of the simulator. Used to restart the simulation
    from a previous step of previous system configuration.

    Args:
        checkpoint_file (str): Name of the file used to store the state_dict periodically.
        every_n_steps (int): Frequency with which checkpoint files are written.
    """

    def __init__(self, checkpoint_file, every_n_steps=1000):
        super(Checkpoint, self).__init__()
        self.every_n_steps = every_n_steps
        self.checkpoint_file = checkpoint_file

    def on_step_end(self, simulator):
        """
        Store state_dict at specified intervals.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        if simulator.step % self.every_n_steps == 0:
            torch.save(simulator.state_dict, self.checkpoint_file)

    def on_simulation_end(self, simulator):
        """
        Store state_dict at the end of the simulation.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        torch.save(simulator.state_dict, self.checkpoint_file)


class DataStream:
    """
    Basic DataStream class to be used with the FileLogger. Creates data groups in the main hdf5 file, accumulates
    the associated information and flushes them to the file periodically.

    Args:
        group_name (str): Name of the data group entry.
    """

    def __init__(self, group_name):
        self.group_name = group_name
        self.buffer = None
        self.data_group = None

        self.main_dataset = None
        self.buffer_size = None
        self.restart = None
        self.every_n_steps = None

    def init_data_stream(
        self, simulator, main_dataset, buffer_size, restart=False, every_n_steps=1
    ):
        """
        Wrapper for initializing the data containers based on the instructions provided in the current simulator. For
        every data stream, the current number of valid entries is stored, which is updated periodically. This is
        necessary if a simulation is e.g. restarted or data is extracted during a running simulations, as all arrays
        are initially constructed taking the full length of the simulation into account.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
            main_dataset (h5py.File): Main h5py dataset object.
            buffer_size (int): Size of the buffer, once full, data is stored to the hdf5 dataset.
            restart (bool): If the simulation is restarted, continue logging in the previously created dataset.
                            (default=False)
            every_n_steps (int): How often simulation steps are logged. Used e.g. to determine overall time step in
                                 MoleculeStream.
        """
        self.main_dataset = main_dataset
        self.buffer_size = buffer_size
        self.restart = restart
        self.every_n_steps = every_n_steps

        self._init_data_stream(simulator)
        # Write number of meaningful entries into attributes
        if not self.restart:
            self.data_group.attrs["entries"] = 0

    def _init_data_stream(self, simulator):
        """
        Specific initialization routine. Needs to be adapted.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        raise NotImplementedError

    def update_buffer(self, buffer_position, simulator):
        """
        Instructions for updating the buffer. Needs to take into account reformatting of data, etc.

        Args:
            buffer_position (int): Current position in the buffer.
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        raise NotImplementedError

    def flush_buffer(self, file_position, buffer_position):
        """
        Write data contained in buffer into the main hdf5 file.

        Args:
            file_position (int): Current position in the main dataset file.
            buffer_position (int): Most recent entry in the buffer. Used to ensure no buffer entries are written to the
                                   main file.
        """
        self.data_group[file_position : file_position + buffer_position] = (
            self.buffer[:buffer_position].detach().cpu()
        )
        # Update number of meaningful entries
        self.data_group.attrs.modify("entries", file_position + buffer_position)
        self.data_group.flush()

    def _setup_data_groups(self, data_shape, simulator):
        """
        Auxiliary routine for initializing data groups in the main hdf5 data file as well as the buffer used during
        logging. All arrays are initialized using the full number of simulation steps specified in the main simulator
        class. The current positions in these arrays are managed via the 'entries' group attribute.

        Args:
            data_shape (list(int)): Shape of the target data tensor
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        # Initialize the buffer
        self.buffer = torch.zeros(
            self.buffer_size, *data_shape, device=simulator.system.device
        )

        if self.restart:
            # Load previous data stream and resize
            self.data_group = self.main_dataset[self.group_name]
            self.data_group.resize(
                (simulator.n_steps + self.data_group.attrs["entries"],) + data_shape
            )
        else:
            # Otherwise, generate new data group in the dataset
            self.data_group = self.main_dataset.create_dataset(
                self.group_name,
                shape=(simulator.n_steps,) + data_shape,
                chunks=self.buffer.shape,
                dtype=np.float32,
                maxshape=(None,) + data_shape,
            )


class PropertyStream(DataStream):
    """
    Main routine for logging the properties predicted by the calculator to the group 'properties' of hdf5 dataset.
    Stores properties in a flattened array and writes names, shapes and positions to the group data section. Since this
    routine determines property shapes based on the system.properties dictionary, at least one computations needs to be
    performed beforehand. Properties are stored in an array of the shape
    n_steps x n_replicas x n_molecules x n_properties, where n_steps is the number of simulation steps, n_replicas and
    n_molecules is the number of simulation replicas and different molecules and n_properties is the length of the
    flattened property array.

    Args:
        target_properties (list): List of properties to be written to the hdf5 database. If no list is given, defaults
                                  to None, which means all properties are stored.
    """

    def __init__(self, target_properties=None):
        super(PropertyStream, self).__init__("properties")
        self.n_replicas = None
        self.n_molecules = None
        self.properties_slices = {}
        self.target_properties = target_properties

    def _init_data_stream(self, simulator):
        """
        Routine for determining the present properties and their respective shapes based on the
        simulator.system.properties dictionary and storing them into the attributes of the hdf5 data group.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules

        if simulator.system.properties is None:
            raise FileLoggerError(
                "Shape of properties could not be determined," "please call calculator"
            )

        # Determine present properties, order and shape thereof
        properties_entries, properties_shape, properties_positions = self._get_properties_structures(
            simulator.system.properties
        )

        data_shape = (self.n_replicas, self.n_molecules, properties_entries)

        self._setup_data_groups(data_shape, simulator)

        if not self.restart:
            # Store metadata on shape and position of properties in array
            self.data_group.attrs["shapes"] = json.dumps(properties_shape)
            self.data_group.attrs["positions"] = json.dumps(properties_positions)

    def update_buffer(self, buffer_position, simulator):
        """
        Routine for updating the propery buffer.

        Args:
            buffer_position (int): Current position in the buffer.
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        # These are already detached in the calculator by default.
        for p in self.properties_slices:
            self.buffer[
                buffer_position : buffer_position + 1, ..., self.properties_slices[p]
            ] = simulator.system.properties[p].view(
                self.n_replicas, self.n_molecules, -1
            )
            # self.buffer[buffer_position:buffer_position + 1, ..., self.properties_slices[p]] = \
            #     simulator.system.properties[p].view(self.n_replicas, self.n_molecules, -1).detach()

    def _get_properties_structures(self, property_dict):
        """
        Auxiliary function to get the names, shapes and positions used in the property stream based on the property
        dictionary of the system.

        Args:
            property_dict (dict(torch.Tensor)): Property dictionary of the main simulator.system class.

        Returns:
            int: Total number of property fields used per replica, molecule and time step.
            dict(slice): Dictionary holding the position of the target property within the flattened array.
            dist(tuple): Dictionary holding the original shapes of the property tensors.
        """
        properties_entries = 0
        properties_shape = {}
        properties_positions = {}

        for p in property_dict:

            if self.target_properties is not None and p not in self.target_properties:
                continue

            # Store shape for metadata
            properties_shape[p] = [int(i) for i in property_dict[p].shape[2:]]
            # Use shape to determine overall array dimensions
            start = properties_entries
            properties_entries += int(np.prod(properties_shape[p]))
            # Get position of property in array
            properties_positions[p] = (start, properties_entries)
            self.properties_slices[p] = slice(start, properties_entries)

        return properties_entries, properties_shape, properties_positions


class SimulationStream(PropertyStream):
    """
    DataStream for dynamic system properties, such as kinetic energy and temperatures of the individual replicas, as
    well as centroids. Adds and updates the group 'simulation'. Stores properties in a flattened array and writes names,
    shapes and positions to the group data section. Properties are stored in an array of the shape
    n_steps x n_replicas x n_molecules x n_properties, where n_steps is the number of simulation steps, n_replicas and
    n_molecules is the number of simulation replicas and different molecules and n_properties is the length of the
    flattened property array.
    """

    def __init__(self):
        super(SimulationStream, self).__init__()
        self.group_name = "simulation"

    def _init_data_stream(self, simulator):
        """
        Get the shape and positions of all monitored properties.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules

        # Create an auxiliary property array from the current system state
        property_dictionary = {
            "kinetic_energy": simulator.system.kinetic_energy,
            "kinetic_energy_centroid": simulator.system.centroid_kinetic_energy,
            "temperature": simulator.system.temperature,
            "temperature_centroid": simulator.system.centroid_temperature,
        }

        properties_entries, properties_shape, properties_positions = self._get_properties_structures(
            property_dictionary
        )

        data_shape = (self.n_replicas, self.n_molecules, properties_entries)

        self._setup_data_groups(data_shape, simulator)

        if not self.restart:
            self.data_group.attrs["shapes"] = json.dumps(properties_shape)
            self.data_group.attrs["positions"] = json.dumps(properties_positions)

    def update_buffer(self, buffer_position, simulator):
        """
        Routine for updating the buffer.

        Args:
            buffer_position (int): Current position in the buffer.
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        property_dictionary = {
            "temperature": simulator.system.temperature,
            "kinetic_energy": simulator.system.kinetic_energy,
            "temperature_centroid": simulator.system.centroid_temperature,
            "kinetic_energy_centroid": simulator.system.centroid_kinetic_energy,
        }

        for p in self.properties_slices:
            self.buffer[
                buffer_position : buffer_position + 1, ..., self.properties_slices[p]
            ] = property_dictionary[p].view(-1, self.n_molecules, 1)


class MoleculeStream(DataStream):
    """
    DataStream for logging atom types, positions and velocities to the group 'molecules' of the main hdf5 dataset.
    Positions and velocities are stored in a n_steps x n_replicas x n_molecules x 6 array, where n_steps is the number
    of simulation steps, n_replicas and n_molecules are the number of simulation replicas and different molecules. The
    first 3 of the final 6 components are the Cartesian positions and the last 3 the velocities in atomic units. Atom
    types, the numbers of replicas, molecules and atoms, as well as the length of the time step in atomic units
    (for spectra) are stored in the group attributes.
    """

    def __init__(self):
        super(MoleculeStream, self).__init__("molecules")
        self.written = 0

    def _init_data_stream(self, simulator):
        """
        Initialize the main data shape and write information on atom types, the numbers of replicas, molecules and
        atoms, as well as the length of the time step in atomic units to the group attributes.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        # Get shape of array depending on whether forces should be stored.
        data_shape = (
            simulator.system.n_replicas,
            simulator.system.n_molecules,
            simulator.system.max_n_atoms,
            6,
        )

        self._setup_data_groups(data_shape, simulator)

        if not self.restart:
            self.data_group.attrs["n_replicas"] = simulator.system.n_replicas
            self.data_group.attrs["n_molecules"] = simulator.system.n_molecules
            self.data_group.attrs["n_atoms"] = simulator.system.n_atoms.cpu()
            self.data_group.attrs["atom_types"] = simulator.system.atom_types.cpu()
            self.data_group.attrs["time_step"] = (
                simulator.integrator.time_step * self.every_n_steps
            )
            self.data_group.attrs["every_n_steps"] = self.every_n_steps

    def update_buffer(self, buffer_position, simulator):
        """
        Routine for updating the buffer.

        Args:
            buffer_position (int): Current position in the buffer.
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        self.buffer[
            buffer_position : buffer_position + 1, ..., :3
        ] = simulator.system.positions
        self.buffer[
            buffer_position : buffer_position + 1, ..., 3:
        ] = simulator.system.velocities


class FileLoggerError(Exception):
    """
    Exception for the FileLogger class.
    """

    pass


class FileLogger(SimulationHook):
    """
    Class for monitoring the simulation and storing the resulting data to a hfd5 dataset. The properties to monitor are
    given via instances of the DataStream class. Uses buffers of a given size, which are accumulated and fushed to the
    main file in regular intervals in order to reduce I/O overhead. All arrays are initialized for the full number of
    requested simulation steps, the current positions in each data group is handled via the 'entries' attribute.

    Args:
        filename (str): Path to the hdf5 database file.
        buffer_size (int): Size of the buffer, once full, data is stored to the hdf5 dataset.
        data_streams list(schnetpack.simulation_hooks.hooks.DataStream): List of DataStreams used to collect and log information
                                                                 to the main hdf5 dataset, default are properties and
                                                                 molecules.
        every_n_steps (int): Frequency with which the buffer is updated.
    """

    def __init__(
        self,
        filename,
        buffer_size,
        data_streams=[MoleculeStream(), PropertyStream()],
        every_n_steps=1,
    ):

        self.every_n_steps = every_n_steps
        self.filename = filename
        self.buffer_size = buffer_size

        # Create an empty variable to hold the HDF5 file upon initialization
        self.file = None

        # Precondition data streams
        self.data_steams = []
        for stream in data_streams:
            self.data_steams += [stream]

        # Counter for file writes
        self.file_position = 0
        self.buffer_position = 0

    def on_simulation_start(self, simulator):
        """
        Initializes all present data streams (creating groups, determining buffer shapes, storing metadata, etc.). In
        addition, the 'entries' attribute of each data stream is read from the existing data set upon restart.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """

        # Flag, if new database should be started or data appended to old one
        append_data = False

        # Check, whether file already exists
        if os.path.exists(self.filename):

            # If file exists and it is the first call of a simulator without restart,
            # raise and error.
            if (not simulator.restart) and (simulator.effective_steps == 0):
                raise FileLoggerError(
                    "File {:s} already exists and simulation was not restarted.".format(
                        self.filename
                    )
                )

            # If either a restart is requested or the simulator has already been called,
            # append to file if it exists.
            if simulator.restart or (simulator.effective_steps > 0):
                append_data = True
        else:
            # If no file is found, automatically generate new one.
            append_data = False

        # Create the HDF5 file
        self.file = h5py.File(self.filename, "a", libver="latest")

        # Construct stream buffers and data groups
        for stream in self.data_steams:
            stream.init_data_stream(
                simulator,
                self.file,
                self.buffer_size,
                restart=append_data,
                every_n_steps=self.every_n_steps,
            )

            # Upon restart, get current position in file
            if append_data:
                self.file_position = stream.data_group.attrs["entries"]

        # Enable single writer, multiple reader flag
        self.file.swmr_mode = True

    def on_step_end(self, simulator):
        """
        Update the buffer of each stream after each specified interval and flush the buffer to the main file if full.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        if simulator.step % self.every_n_steps == 0:
            # If buffers are full, write to file
            if self.buffer_position == self.buffer_size:
                self._write_buffer()

            # Update stream buffers
            for stream in self.data_steams:
                stream.update_buffer(self.buffer_position, simulator)

            self.buffer_position += 1

    def on_simulation_end(self, simulator):
        """
        Perform one final flush of the buffers and close the file upon the end of the simulation.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        # Flush remaining data in buffer
        if self.buffer_position != 0:
            self._write_buffer()

        # Close database file
        self.file.close()

    def _write_buffer(self):
        """
        Write all current buffers to the database file.
        """
        for stream in self.data_steams:
            stream.flush_buffer(self.file_position, self.buffer_position)

        self.file_position += self.buffer_size
        self.buffer_position = 0


class TensorboardLogger(SimulationHook):
    """
    Class for logging scalar information of the system replicas and molecules collected during the simulation to
    TensorBoard. An individual scalar is created for every molecule, replica and property.

    Args:
        log_file (str): Path to the TensorBoard file.
        every_n_steps (int): Frequency with which data is logged to TensorBoard.
    """

    def __init__(self, log_file, every_n_steps=100):
        from tensorboardX import SummaryWriter

        self.log_file = log_file
        self.every_n_steps = every_n_steps
        self.writer = SummaryWriter(self.log_file)

        self.n_replicas = None
        self.n_molecules = None

    def on_simulation_start(self, simulator):
        """
        Extract the number of molecules and replicas from simulator.system upon simulation start.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules

    def on_step_end(self, simulator):
        """
        Routine for collecting and storing scalar properties of replicas and molecules during the simulation. Needs to
        be adapted based on the properties.
        In the easiest case, information on group names, etc. is passed to the self._log_group auxiliary function.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        raise NotImplementedError

    def _log_group(self, group_name, step, property, property_centroid=None):
        """
        Auxiliary routine for logging the scalar data associated with the target property. An individual entry is
        created for every replica and molecule. If requested, an entry corresponding to the systems centroid is also
        created.

        Args:
            group_name (str): Base name of the property group to log.
            step (int): Current simulation step.
            property (torch.Tensor): Tensor of the shape (n_replicas x n_molecules) holding the scalar properties of
                                     each replica and molecule.
            property_centroid (torch.Tensor): Also store the centroid of the monitored property if provided
                                              (default=None).
        """
        logger_dict = {}

        for molecule in range(self.n_molecules):
            mol_name = "{:s}/molecule_{:02d}".format(group_name, molecule + 1)

            if property_centroid is not None:
                logger_dict["centroid"] = property_centroid[0, molecule]

            for replica in range(self.n_replicas):
                rep_name = "r{:02d}".format(replica + 1)
                logger_dict[rep_name] = property[replica, molecule]

            self.writer.add_scalars(mol_name, logger_dict, step)

    def on_simulation_end(self, simulator):
        """
        Close the TensorBoard logger.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        self.writer.close()


class TemperatureLogger(TensorboardLogger):
    """
    TensorBoard logging hook for the temperatures of the replicas, as well as of the corresponding centroids for each
    molecule in the system container.

    Args:
        log_file (str): Path to the TensorBoard file.
        every_n_steps (int): Frequency with which data is logged to TensorBoard.
    """

    def __init__(self, log_file, every_n_steps=100):
        super(TemperatureLogger, self).__init__(log_file, every_n_steps=every_n_steps)

    def on_step_end(self, simulator):
        """
        Log the systems temperatures at the given intervals.

        Args:
            simulator (schnetpack.simulation_hooks.Simulator): Simulator class used in the molecular dynamics simulation.
        """
        if simulator.step % self.every_n_steps == 0:
            # Use the _log_group routine to log the systems temperatures
            self._log_group(
                "temperature",
                simulator.step,
                simulator.system.temperature,
                property_centroid=simulator.system.centroid_temperature,
            )
