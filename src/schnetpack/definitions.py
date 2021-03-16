__all__ = ["Structure", "Properties"]


class Structure:
    """
    Keys to access structure properties
    """

    idx = "_idx"
    Z = "_atomic_numbers"
    position = "_positions"
    R = position

    cell = "_cell"
    cell_offset = "_cell_offset"
    pbc = "_pbc"

    seg_m = "_seg_m"
    idx_i = "_idx_i"
    idx_j = "_idx_j"
    Rij = "_Rij"
    n_atoms = "_n_atoms"

    # neighbor_pairs_j = "_neighbor_pairs_j"
    # neighbor_pairs_k = "_neighbor_pairs_k"
    # neighbor_offsets_j = "_neighbor_offsets_j"
    # neighbor_offsets_k = "_neighbor_offsets_k"

    # neighbors_lr = "_neighbors_lr"
    # neighbor_mask_lr = "_neighbor_mask_lr"
    # cell_offset_lr = "_cell_offset_lr"


class Properties:
    # chemical properties
    energy = "energy"
    forces = "forces"
    stress = "stress"
    dipole_moment = "dipole_moment"
    # total_dipole_moment = "total_dipole_moment"
    # polarizability = "polarizability"
    # iso_polarizability = "iso_polarizability"
    # at_polarizability = "at_polarizability"
    # charges = "charges"
    # energy_contributions = "energy_contributions"
    # shielding = "shielding"
    # hessian = "hessian"
    # dipole_derivatives = "dipole_derivatives"
    # polarizability_derivatives = "polarizability_derivatives"
    # electric_field = "electric_field"
    # magnetic_field = "magnetic_field"
    # dielectric_constant = "dielectric_constant"
    # magnetic_moments = "magnetic_moments"
    #
    # properties = [
    #     energy,
    #     forces,
    #     stress,
    #     dipole_moment,
    #     polarizability,
    #     shielding,
    #     hessian,
    #     dipole_derivatives,
    #     polarizability_derivatives,
    #     electric_field,
    #     magnetic_field,
    # ]
    #
    # external_fields = [electric_field, magnetic_field]
    #
    # electric_properties = [
    #     dipole_moment,
    #     dipole_derivatives,
    #     dipole_derivatives,
    #     polarizability_derivatives,
    #     polarizability,
    # ]
    # magnetic_properties = [shielding]
    #
    # required_grad = {
    #     energy: [],
    #     forces: [position],
    #     hessian: [position],
    #     dipole_moment: [electric_field],
    #     polarizability: [electric_field],
    #     dipole_derivatives: [electric_field, position],
    #     polarizability_derivatives: [electric_field, position],
    #     shielding: [magnetic_field, magnetic_moments],
    # }
