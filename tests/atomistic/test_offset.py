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
    remove_offsets = RemoveOffsets(
        property=property,
        remove_mean=use_mean,
        remove_atomrefs=use_atomrefs,
        atomrefs=atomrefs,
        property_mean=mean,
    )

    add_offsets = AddOffsets(
        property=property,
        add_mean=use_mean,
        add_atomrefs=use_atomrefs,
        atomrefs=atomrefs,
        property_mean=mean,
    )

    return remove_offsets, add_offsets


def run_transform_test(property, use_mean, use_atomrefs, mean=None, atomrefs=None):
    """
    Run a test case with specified transform parameters.
    """
    inputs = get_dummy_inputs()
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
    mean = torch.tensor([-20.0])

    # Create atomrefs tensor
    atomrefs_values = [0.0] * 9
    atomrefs_values[1] = -0.5  # H
    atomrefs_values[6] = -1.0  # C
    atomrefs_values[8] = -1.5  # O
    atomrefs_tensor = torch.tensor(atomrefs_values, dtype=torch.float32)

    print("--------------------------------")
    print(f"\nTesting uses_mean only without mean\n")
    run_transform_test(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=None,
        atomrefs=None,
    )

    print("--------------------------------")
    print(f"\nTesting mean only with mean\n")
    run_transform_test(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=mean,
        atomrefs=None,
    )

    print("--------------------------------")
    print(f"\nTesting use_atomrefs only with None\n")
    run_transform_test(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=None,
    )

    print("--------------------------------")
    print(f"\nTesting use_atomrefs only with atomrefs tensor\n")
    run_transform_test(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
    )

    print("--------------------------------")
    print(f"\nTesting both mean and atomrefs with None\n")
    run_transform_test(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=None,
        atomrefs=None,
    )

    print("--------------------------------")
    print(f"\nTesting both use_mean and use_atomrefs with mean and atomrefs\n")
    run_transform_test(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=mean,
        atomrefs=atomrefs_tensor,
    )


if __name__ == "__main__":
    test_all_combinations()
