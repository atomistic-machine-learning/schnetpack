import os
import torch
import numpy as np
import json
import h5py

__all__ = ['Checkpoint', 'RemoveCOMMotion', 'BiasPotential', 'FileLogger',
           'TensorboardLogger', 'TemperatureLogger', 'MoleculeStream',
           'PropertyStream']


class SimulationHook:
    """
    Basic class for simulator hooks
    """

    @property
    def state_dict(self):
        return {}

    @state_dict.setter
    def state_dict(self, state_dict):
        pass

    def on_step_begin(self, simulator):
        pass

    def on_step_middle(self, simulator):
        pass

    def on_step_end(self, simulator):
        pass

    def on_step_failed(self, simulator):
        pass

    def on_simulation_start(self, simulator):
        pass

    def on_simulation_end(self, simulator):
        pass


class Checkpoint(SimulationHook):

    def __init__(self, checkpoint_file, every_n_steps=1000):
        super(Checkpoint, self).__init__()
        self.every_n_steps = every_n_steps
        self.checkpoint_file = checkpoint_file

    def on_step_end(self, simulator):
        if simulator.step % self.every_n_steps == 0:
            torch.save(simulator.state_dict, self.checkpoint_file)

    def on_simulation_end(self, simulator):
        torch.save(simulator.state_dict, self.checkpoint_file)


class RemoveCOMMotion(SimulationHook):

    def __init__(self, every_n_steps=10, remove_rotation=True):
        self.every_n_steps = every_n_steps
        self.remove_rotation = remove_rotation

    def on_step_end(self, simulator):
        if simulator.step % self.every_n_steps == 0:
            simulator.system.remove_com()
            simulator.system.remove_com_translation()
            if self.remove_rotation:
                simulator.system.remove_com_rotation()


class BiasPotential(SimulationHook):

    def __init__(self):
        raise NotImplementedError

    def on_step_end(self, simulator):
        raise NotImplementedError


class DataStream:

    def __init__(self, group_name, main_dataset, buffer_size, restart=False):
        self.group_name = group_name
        self.main_dataset = main_dataset
        self.buffer_size = buffer_size
        self.restart = restart

        self.buffer = None
        self.data_group = None

    def init_data_stream(self, simulator):
        self._init_data_stream(simulator)
        # Write number of meaningful entries into attributes
        if not self.restart:
            self.data_group.attrs['entries'] = 0

    def update_buffer(self, buffer_position, simulator):
        raise NotImplementedError

    def flush_buffer(self, file_position, buffer_position):
        self.data_group[file_position:file_position + buffer_position] = \
            self.buffer[:buffer_position].cpu()
        # Update number of meaningful entries
        self.data_group.attrs.modify('entries', file_position + buffer_position)
        self.data_group.flush()

    def _init_data_stream(self, simulator):
        raise NotImplementedError

    def _setup_data_groups(self, data_shape, simulator):
        # Initialize the buffer
        self.buffer = torch.zeros(self.buffer_size, *data_shape,
                                  device=simulator.system.device)

        if self.restart:
            # Load previous data stream and resize
            self.data_group = self.main_dataset[self.group_name]
            self.data_group.resize((simulator.n_steps
                                    + self.data_group.attrs['entries'],)
                                   + data_shape)
        else:
            # Otherwise, generate new data group in the dataset
            self.data_group = self.main_dataset.create_dataset(
                self.group_name, shape=(simulator.n_steps,) + data_shape,
                chunks=self.buffer.shape, dtype=np.float32,
                maxshape=(None,) + data_shape)


class PropertyStream(DataStream):

    def __init__(self, main_dataset, buffer_size, restart=False):
        super(PropertyStream, self).__init__('properties', main_dataset,
                                             buffer_size, restart=restart)

        self.n_replicas = None
        self.n_molecules = None
        self.properties_slices = {}

    def _init_data_stream(self, simulator):

        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules

        if simulator.system.properties is None:
            raise FileLoggerError('Shape of properties could not be determined,'
                                  'please call calculator')

        # Determine present properties, order and shape thereof
        properties_entries, properties_shape, properties_positions = \
            self._get_properties_structures(simulator.system.properties)

        data_shape = (self.n_replicas, self.n_molecules, properties_entries)

        self._setup_data_groups(data_shape, simulator)

        if not self.restart:
            # Store metadata on shape and position of properties in array
            self.data_group.attrs['shapes'] = json.dumps(properties_shape)
            self.data_group.attrs['positions'] = \
                json.dumps(properties_positions)

    def update_buffer(self, buffer_position, simulator):
        for p in self.properties_slices:
            self.buffer[buffer_position:buffer_position + 1, ..., self.properties_slices[p]] = \
                simulator.system.properties[p].view(self.n_replicas, self.n_molecules, -1).detach()

    def _get_properties_structures(self, property_dict):
        """
        Auxiliary function to generate data structures from property dictionary
        """
        properties_entries = 0
        properties_shape = {}
        properties_positions = {}

        for p in property_dict:
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

    def __init__(self, main_dataset, buffer_size, restart=False):
        super(SimulationStream, self).__init__(main_dataset, buffer_size,
                                               restart=restart)

        self.group_name = 'simulation'

    def _init_data_stream(self, simulator):
        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules

        property_dictionary = {
            'kinetic_energy': simulator.system.kinetic_energy,
            'kinetic_energy_centroid': simulator.system.centroid_kinetic_energy,
            'temperature': simulator.system.temperature,
            'temperature_centroid': simulator.system.centroid_temperature
        }

        properties_entries, properties_shape, properties_positions = \
            self._get_properties_structures(property_dictionary)

        data_shape = (self.n_replicas, self.n_molecules, properties_entries)

        self._setup_data_groups(data_shape, simulator)

        if not self.restart:
            self.data_group.attrs['shapes'] = json.dumps(properties_shape)
            self.data_group.attrs['positions'] = \
                json.dumps(properties_positions)

    def update_buffer(self, buffer_position, simulator):
        property_dictionary = {
            'temperature': simulator.system.temperature,
            'kinetic_energy': simulator.system.kinetic_energy,
            'temperature_centroid': simulator.system.centroid_temperature,
            'kinetic_energy_centroid': simulator.system.centroid_kinetic_energy
        }

        for p in self.properties_slices:
            self.buffer[buffer_position:buffer_position + 1, ..., self.properties_slices[p]] = \
                property_dictionary[p].view(-1, self.n_molecules, 1)


class MoleculeStream(DataStream):

    def __init__(self, main_dataset, buffer_size, restart=False):
        super(MoleculeStream, self).__init__('molecules', main_dataset,
                                             buffer_size, restart=restart)
        self.written = 0

    def _init_data_stream(self, simulator):
        data_shape = (simulator.system.n_replicas, simulator.system.n_molecules,
                      simulator.system.max_n_atoms, 6)

        self._setup_data_groups(data_shape, simulator)

        if not self.restart:
            self.data_group.attrs['n_replicas'] = simulator.system.n_replicas
            self.data_group.attrs['n_molecules'] = simulator.system.n_molecules
            self.data_group.attrs['n_atoms'] = simulator.system.n_atoms.cpu()
            self.data_group.attrs['atom_types'] = simulator.system.atom_types.cpu()
            self.data_group.attrs['time_step'] = simulator.integrator.time_step

    def update_buffer(self, buffer_position, simulator):
        self.buffer[buffer_position:buffer_position + 1, ..., :3] = \
            simulator.system.positions
        self.buffer[buffer_position:buffer_position + 1, ..., 3:] = \
            simulator.system.velocities


class FileLoggerError(Exception):
    pass


class FileLogger(SimulationHook):

    def __init__(self, filename, buffer_size,
                 data_streams=[MoleculeStream, PropertyStream],
                 every_n_steps=1, restart=False):

        self.restart = restart
        self.every_n_steps = every_n_steps

        # Remove already existing file if not restarting simulation
        if not self.restart:
            if os.path.exists(filename):
                os.remove(filename)

        self.file = h5py.File(filename, 'a', libver='latest')
        self.buffer_size = buffer_size

        # Precondition data streams
        self.data_steams = []
        for stream in data_streams:
            self.data_steams += [stream(self.file, self.buffer_size,
                                        restart=self.restart)]

        # Counter for file writes
        self.file_position = 0
        self.buffer_position = 0

    def on_simulation_start(self, simulator):
        # Construct stream buffers and data groups
        for stream in self.data_steams:
            stream.init_data_stream(simulator)
            # Upon restart, get current position in file
            # TODO: This is not the nicest way, but h5py does not seem to
            #  support life update of metadata
            if self.restart:
                self.file_position = stream.data_group.attrs['entries']

        # Enable single writer, multiple reader flag
        self.file.swmr_mode = True

    def on_step_end(self, simulator):
        if simulator.step % self.every_n_steps == 0:
            # If buffers are full, write to file
            if self.buffer_position == self.buffer_size:
                self._write_buffer()

            # Update stream buffers
            for stream in self.data_steams:
                stream.update_buffer(self.buffer_position, simulator)

            self.buffer_position += 1

    def on_simulation_end(self, simulator):
        # Flush remaining data in buffer
        if self.buffer_position != 0:
            self._write_buffer()

        # Close database file
        self.file.close()

    def _write_buffer(self):
        for stream in self.data_steams:
            stream.flush_buffer(self.file_position, self.buffer_position)

        self.file_position += self.buffer_size
        self.buffer_position = 0


class TensorboardLogger(SimulationHook):

    def __init__(self, log_file, every_n_steps=100):
        from tensorboardX import SummaryWriter
        self.log_file = log_file
        self.every_n_steps = every_n_steps
        self.writer = SummaryWriter(self.log_file)

        self.n_replicas = None
        self.n_molecules = None

    def on_simulation_start(self, simulator):
        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules

    def on_step_end(self, simulator):
        raise NotImplementedError

    def _log_group(self, group_name, step, property, property_centroid=None):

        logger_dict = {}

        for molecule in range(self.n_molecules):
            mol_name = '{:s}/molecule_{:02d}'.format(group_name, molecule + 1)

            if property_centroid is not None:
                logger_dict['centroid'] = property_centroid[0, molecule]

            for replica in range(self.n_replicas):
                rep_name = 'r{:02d}'.format(replica + 1)
                logger_dict[rep_name] = property[replica, molecule]

            self.writer.add_scalars(mol_name, logger_dict, step)

    def on_simulation_end(self, simulator):
        self.writer.close()


class TemperatureLogger(TensorboardLogger):

    def __init__(self, log_file, every_n_steps=100):
        super(TemperatureLogger, self).__init__(log_file,
                                                every_n_steps=every_n_steps)

    def on_step_end(self, simulator):
        if simulator.step % self.every_n_steps == 0:
            self._log_group(
                'temperature',
                simulator.step,
                simulator.system.temperature,
                property_centroid=simulator.system.centroid_temperature)
