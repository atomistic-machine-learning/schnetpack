from schnetpack.data import Structure
from schnetpack.md.utils import MDUnits


class MDCalculatorError(Exception):
    pass


class MDCalculator:

    def __init__(self, required_properties,
                 force_handle,
                 position_conversion=1.0,
                 force_conversion=1.0,
                 property_conversion={}):
        self.results = {}
        self.force_handle = force_handle
        print('*', required_properties)
        self.required_properties = required_properties
        self.position_conversion = position_conversion
        self.force_conversion = force_conversion

        self.property_conversion = property_conversion
        self._init_default_conversion()

    def calculate(self, system):
        raise NotImplementedError

    def _init_default_conversion(self):
        """
        Set default conversion factors
        """
        for p in self.required_properties:
            if p not in self.property_conversion:
                self.property_conversion[p] = 1.0

    def _update_system(self, system):
        """
        Convert, reformat and set properties of the system. Forces are
        treated separately.
        """
        for p in self.required_properties:
            print(self.results.keys())
            print(self.required_properties)
            if p not in self.results:
                raise MDCalculatorError('Requested property {:s} not in '
                                        'results'.format(p))
            elif p == self.force_handle:
                self._set_system_forces(system)
            else:
                dim = self.results[p].shape
                system.properties[p] = self.results[p].view(
                    system.n_replicas, system.n_molecules, *dim[1:]) * \
                    self.property_conversion[p]

    def _get_system_neighbors(self, system):
        """
        Auxiliary function to get properly formatted neighbor lists from
        system class
        """
        if system.neighbor_list is None:
            raise ValueError('System does not have neighbor list.')
        neighbor_list, neighbor_mask = system.neighbor_list.get_neighbors()

        neighbor_list = neighbor_list.view(-1, system.max_n_atoms,
                                           system.max_n_atoms - 1)
        neighbor_mask = neighbor_mask.view(-1, system.max_n_atoms,
                                           system.max_n_atoms - 1)
        return neighbor_list, neighbor_mask

    def _get_system_molecules(self, system):
        """
        Auxiliary function to get properly formatted tensors from system class
        """
        positions = system.positions.view(-1, system.max_n_atoms,
                                          3) * self.position_conversion

        atom_types = system.atom_types.view(-1, system.max_n_atoms)
        atom_masks = system.atom_masks.view(-1, system.max_n_atoms)
        return positions, atom_types, atom_masks

    def _set_system_forces(self, system):
        """
        Auxiliary function to properly set and format system forces.
        """
        forces = self.results[self.force_handle]
        system.forces = forces.view(system.n_replicas, system.n_molecules,
                                    system.max_n_atoms,3) *  \
                        self.force_conversion

    def _get_ase_molecules(self, system):
        pass


class SchnetPackCalculator(MDCalculator):

    def __init__(self, model, required_properties,
                 force_handle,
                 position_conversion=1.0 / MDUnits.angs2bohr,
                 force_conversion=1.0 / MDUnits.auforces2aseforces,
                 property_conversion={}):
        super(SchnetPackCalculator, self).__init__(required_properties,
                                                   force_handle,
                                                   position_conversion,
                                                   force_conversion,
                                                   property_conversion)

        self.model = model

    def calculate(self, system):
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _generate_input(self, system):
        positions, atom_types, atom_masks = self._get_system_molecules(system)
        neighbors, neighbor_mask = self._get_system_neighbors(system)

        inputs = {
            Structure.R: positions,
            Structure.Z: atom_types,
            Structure.atom_mask: atom_masks,
            Structure.cell: None,
            Structure.cell_offset: None,
            Structure.neighbors: neighbors,
            Structure.neighbor_mask: neighbor_mask
        }

        return inputs
