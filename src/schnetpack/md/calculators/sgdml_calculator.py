from schnetpack.md.calculators import MDCalculator
from schnetpack.md.utils import MDUnits
from schnetpack import Properties


class SGDMLCalculator(MDCalculator):
    """
    Calculator for the sGDML model published in [#sgdml1]_ and [#sgdml2]_ . This model predicts energies and forces and
    currently only is available for molecules of the same size. In order to use the calculator, the sgdml code package
    available online at https://github.com/stefanch/sGDML and described in [#sgdml3]_ is required.

    Args:
        model (torch.nn.module): Loaded sGDML model.
        required_properties (list): Properties to predict with sGDML, available are energies and forces, which are
                                    default.
        force_handle (str): Name of forces in output. Default is 'forces'.
        force_conversion (float): Convert forces from sGDML to atomic units. sGDML used kcal/mol/Angstrom, so this
                                  default is used here.
        position_conversion (float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177. Since sGDML uses Angstrom and
                             the MD algorithm atomic units, this factor is used as default here
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        detach (bool): Detach property computation graph after every calculator call. Enabled by default. Should only
                       be disabled if one wants to e.g. compute derivatives over short trajectory snippets.

    References
    ----------
    .. [#sgdml1] Chmiela, Tkatchenko, Sauceda, Poltavsky, Schütt, Müller:
       Energy-conserving Molecular Force Fields.
       Science Advances, 3 (5), e1603015. 2017.
    .. [#sgdml2] Chmiela, Sauceda, Müller, Tkatchenko:
       Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.
       Nature Communications, 9 (1), 3887. 2018.
    .. [#sgdml3] Chmiela, Sauceda, Poltavsky, Müller, Tkatchenko:
       sGDML: Constructing accurate and data efficient molecular force fields using machine learning.
       Computer Physics Communications (in press). https://doi.org/10.1016/j.cpc.2019.02.007
    """

    def __init__(
        self,
        model,
        required_properties=[Properties.energy, Properties.forces],
        force_handle=Properties.forces,
        position_conversion=1.0 / MDUnits.angs2bohr,
        force_conversion=1.0 / MDUnits.Ha2kcalpmol / MDUnits.angs2bohr,
        property_conversion={},
        detach=True,
    ):
        super(SGDMLCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion,
            force_conversion,
            property_conversion,
            detach,
        )

        self.model = model

    def calculate(self, system):
        """
        Main routine, extract the positions from the current system, use the sGDML model to predict the energies and
        forces and construct the results dictionary, which is then used to update the system.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        energy, forces = self.model(inputs)

        self.results = {Properties.energy: energy, Properties.forces: forces}
        self._update_system(system)

    def _generate_input(self, system):
        """
        Function to extracts atom_types, positions and atom_masks from the system and generate a properly
        formatted input for sGDML, which in this case is just the array of positions.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            torch.Tensor: sGDML inputs, which are the positions in the format N_replicas*N_molecules x N_atoms x 3.
        """
        positions, atom_types, atom_masks = self._get_system_molecules(system)

        return positions
