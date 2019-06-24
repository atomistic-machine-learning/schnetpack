class Structure:
    """
    Keys to access structure properties in `schnetpack.data.AtomsData`
    """

    Z = "_atomic_numbers"
    charge = "_charge"
    atom_mask = "_atom_mask"
    R = "_positions"
    cell = "_cell"
    pbc = "_pbc"
    neighbors = "_neighbors"
    neighbor_mask = "_neighbor_mask"
    cell_offset = "_cell_offset"
    neighbor_pairs_j = "_neighbor_pairs_j"
    neighbor_pairs_k = "_neighbor_pairs_k"
    neighbor_pairs_mask = "_neighbor_pairs_mask"


class Properties:
    energy = 'energy'
    forces = 'forces'
    dipole_moment = 'dipole_moment'
    polarizability = 'polarizability'
    shielding = 'shielding'
    hessian = 'hessian'
    dipole_derivatives = 'dipole_derivatives'
    polarizability_derivatives = 'polarizability_derivatives'
    electric_field = 'electric_field'
    magnetic_field = 'magnetic_field'
    dielectric_constant = 'dielectric_constant'
    magnetic_moments = 'magnetic_moments'
    position = Structure.R

    properties = [
        energy,
        forces,
        dipole_moment,
        polarizability,
        shielding,
        hessian,
        dipole_derivatives,
        polarizability_derivatives,
        electric_field,
        magnetic_field
    ]

    external_fields = [
        electric_field,
        magnetic_field
    ]

    electric_properties = [dipole_moment, dipole_derivatives, dipole_derivatives, polarizability_derivatives,
                           polarizability]
    magnetic_properties = [shielding]

    required_grad = {
        energy: [],
        forces: [position],
        hessian: [position],
        dipole_moment: [electric_field],
        polarizability: [electric_field],
        dipole_derivatives: [electric_field, position],
        polarizability_derivatives: [electric_field, position],
        shielding: [magnetic_field, magnetic_moments]
    }
