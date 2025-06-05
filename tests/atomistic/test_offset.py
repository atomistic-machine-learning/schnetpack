import torch
import schnetpack.properties as structure
from schnetpack.transform import AddOffsets, RemoveOffsets


def get_dummy_inputs():
    """
    Structure:
    - Ethanol: H3C-CH2-OH (9 atoms)
    - Methanol: H3C-OH (6 atoms)
    - Water: H2O (3 atoms)
    """
    ethanol_Z = [6, 1, 1, 1, 6, 1, 1, 8, 1]  # CH3CH2OH
    methanol_Z = [6, 1, 1, 1, 8, 1]  # CH3OH
    water_Z = [8, 1, 1]  # H2O

    all_Z = ethanol_Z + methanol_Z + water_Z

    # Create molecule indices
    ethanol_idx = [0] * len(ethanol_Z)
    methanol_idx = [1] * len(methanol_Z)
    water_idx = [2] * len(water_Z)
    all_idx = ethanol_idx + methanol_idx + water_idx

    n_atoms = torch.tensor([len(ethanol_Z), len(methanol_Z), len(water_Z)])

    # Random energy for each molecule
    energies = torch.tensor([-170.0, -115.0, -76.0])  # ethanol, methanol, water

    inputs = {
        structure.Z: torch.tensor(all_Z),
        structure.n_atoms: n_atoms,
        structure.idx_m: torch.tensor(all_idx),
        "energy": energies,
    }

    return inputs


def get_offset_transforms(property, use_mean, use_atomrefs, mean=None, atomrefs=None):
    """
    Create RemoveOffsets and AddOffsets transforms with specified parameters.

    Args:
        property (str): Property to transform
        use_mean (bool): Whether to use mean offset
        use_atomrefs (bool): Whether to use atomrefs
        mean (torch.Tensor): Mean value for the property
        atomrefs (torch.Tensor): Atom reference values
    """
    atomrefs_tensor = None
    if atomrefs is not None:
        atomrefs_tensor = torch.tensor(atomrefs, dtype=torch.float32)

    remove_offsets = RemoveOffsets(
        property=property,
        remove_mean=use_mean,
        remove_atomrefs=use_atomrefs,
        atomrefs=atomrefs_tensor,
        property_mean=mean if use_mean else None,
    )

    add_offsets = AddOffsets(
        property=property,
        add_mean=use_mean,
        add_atomrefs=use_atomrefs,
        atomrefs=atomrefs_tensor,
        property_mean=mean if use_mean else None,
    )

    return remove_offsets, add_offsets


def run_transform_test(property, use_mean, use_atomrefs, mean=None, atomrefs=None):
    """
    Run a test case with specified transform parameters.
    """
    inputs = get_dummy_inputs()
    if atomrefs is not None:
        atomrefs = torch.tensor(atomrefs, dtype=torch.float32)
        print(f"Atomrefs tensor: {atomrefs}")
        print(f"Atomrefs tensor shape: {atomrefs.shape}")

    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=use_mean,
        use_atomrefs=use_atomrefs,
        mean=mean,
        atomrefs=atomrefs,
    )

    print("\nInput molecules:")
    print(f"Ethanol: {inputs[property][0]}")
    print(f"nMethanol: {inputs[property][1]}")
    print(f"Water: {inputs[property][2]}")

    intermediate = remove_offsets(inputs.copy())
    final = add_offsets(intermediate)

    is_equal = torch.allclose(final[property], inputs[property])

    print("\nTransform results:")
    print("After remove_offsets:")
    for i, mol in enumerate(["Ethanol", "Methanol", "Water"]):
        print(f"{mol}: {intermediate[property][i]}")

    print("\nAfter add_offsets:")
    for i, mol in enumerate(["Ethanol", "Methanol", "Water"]):
        print(f"{mol}: {final[property][i]}")

    print("\nTransforms are inverse operations:", is_equal)


def test_all_combinations():
    """
    Test all meaningful combinations of transform parameters.
    """
    property = "energy"
    mean_values = [torch.tensor([-20.0])]
    atomrefs_values = [[0.0, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.5]]

    # Test all combinations
    for mean in mean_values:
        for atomrefs in atomrefs_values:
            print(f"Testing uses_mean only without mean")
            run_transform_test(
                property=property,
                use_mean=True,
                use_atomrefs=False,
                mean=None,
                atomrefs=None,
            )
            print(f"Testing mean only with mean")
            run_transform_test(
                property=property,
                use_mean=True,
                use_atomrefs=False,
                mean=mean,
                atomrefs=None,
            )

            print(f"Testing use_atomrefs only with None")
            run_transform_test(
                property=property,
                use_mean=False,
                use_atomrefs=True,
                mean=None,
                atomrefs=None,
            )

            print(f"Testing use_atomrefs only with atomrefs tensor")
            run_transform_test(
                property=property,
                use_mean=False,
                use_atomrefs=True,
                mean=None,
                atomrefs=atomrefs,
            )

            print(f"Testing both mean and atomrefs with None")
            run_transform_test(
                property=property,
                use_mean=True,
                use_atomrefs=True,
                mean=None,
                atomrefs=None,
            )
            print(f"Testing both use_mean and use_atomrefs with mean and atomrefs")
            run_transform_test(
                property=property,
                use_mean=True,
                use_atomrefs=True,
                mean=mean,
                atomrefs=atomrefs,
            )


if __name__ == "__main__":
    test_all_combinations()
