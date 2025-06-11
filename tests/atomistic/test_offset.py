import pytest
import torch
import schnetpack.properties as structure
from schnetpack.transform import AddOffsets, RemoveOffsets


# === Fixtures ===
@pytest.fixture
def dummy_inputs():
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

    # For extensive properties, n_atoms is per molecule
    n_atoms = torch.tensor([len(ethanol_Z), len(methanol_Z), len(water_Z)])
    # Random energy for each molecule
    energies = torch.tensor([-170.0, -115.0, -76.0])

    inputs = {
        structure.Z: torch.tensor(all_Z),
        structure.n_atoms: n_atoms,
        structure.idx_m: torch.tensor(all_idx),
        "energy": energies,
    }

    return inputs


@pytest.fixture
def atomrefs_tensor():
    """Create atomrefs tensor with proper size (zmax=100)"""
    atomrefs_values = [0.0] * 100
    atomrefs_values[1] = -0.5  # H
    atomrefs_values[6] = -1.0  # C
    atomrefs_values[8] = -1.5  # O
    return torch.tensor(atomrefs_values, dtype=torch.float32)


@pytest.fixture
def mean_tensor():
    """Create mean tensor for testing"""
    return torch.tensor([-20.0])


# === Helper Functions ===
def get_offset_transforms(
    property, use_mean, use_atomrefs, mean=None, atomrefs=None, is_extensive=True
):
    """
    Create RemoveOffsets and AddOffsets transforms with specified parameters.

    Args:
        property (str): Property to transform
        use_mean (bool): Whether to use mean offset
        use_atomrefs (bool): Whether to use atomrefs
        mean (torch.Tensor): Mean value for the property
        atomrefs (torch.Tensor): Atom reference values
        is_extensive (bool): Whether the property is extensive
    """
    remove_offsets = RemoveOffsets(
        property=property,
        remove_mean=use_mean,
        remove_atomrefs=use_atomrefs,
        atomrefs=atomrefs,
        property_mean=mean,
        is_extensive=is_extensive,
    )

    add_offsets = AddOffsets(
        property=property,
        add_mean=use_mean,
        add_atomrefs=use_atomrefs,
        atomrefs=atomrefs,
        property_mean=mean,
        is_extensive=is_extensive,
    )

    return remove_offsets, add_offsets


# === Test Functions ===
def test_remove_add_offsets_mean_none_extensive(dummy_inputs):
    """Test use_mean with None and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=None,
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_mean_extensive(dummy_inputs, mean_tensor):
    """Test use_mean with mean_tensor and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=mean_tensor,
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_mean_none_intensive(dummy_inputs):
    """Test use_mean with None and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=None,
        atomrefs=None,
        is_extensive=False,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_mean_intensive(dummy_inputs, mean_tensor):
    """Test use_mean with mean_tensor and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=mean_tensor,
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_atomrefs_none_extensive(dummy_inputs):
    """Test use_atomrefs None and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_atomrefs_tensor_extensive(dummy_inputs, atomrefs_tensor):
    """Test use_atomrefs tensor and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_atomrefs_none(dummy_inputs):
    """Test use_atomrefs None and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=None,
        is_extensive=False,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_atomrefs_tensor(dummy_inputs, atomrefs_tensor):
    """Test use_atomrefs tensor and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_both_mean_and_atomrefs_none_extensive(dummy_inputs):
    """Test both mean and atomrefs with none and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=None,
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_both_mean_and_atomrefs_extensive(
    dummy_inputs, mean_tensor, atomrefs_tensor
):
    """Test both mean and atomrefs with tensors and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=mean_tensor,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_both_mean_and_atomrefs_none(dummy_inputs):
    """Test both mean and atomrefs with none and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=None,
        atomrefs=None,
        is_extensive=False,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_both_mean_and_atomrefs_intensive(
    dummy_inputs, mean_tensor, atomrefs_tensor
):
    """Test both mean and atomrefs with tensors and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=mean_tensor,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    print("Intermediate:", intermediate[property])
    final = add_offsets(intermediate.copy())
    print("Final:", final[property])

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_no_transforms_extensive(dummy_inputs):
    """Test RemoveOffsets and AddOffsets with no transforms and extensive True"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=False,
        mean=None,
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_add_offsets_no_transforms_intensive(dummy_inputs):
    """Test RemoveOffsets and AddOffsets with no transforms and extensive False"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=False,
        mean=None,
        atomrefs=None,
        is_extensive=False,
    )

    intermediate = remove_offsets(dummy_inputs.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs[property])


def test_remove_offsets_preserves_other_properties(dummy_inputs):
    """Test that RemoveOffsets preserves other properties in the input"""
    property = "energy"
    remove_offsets, _ = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=torch.tensor([-20.0]),
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = remove_offsets(dummy_inputs.copy())

    for key in dummy_inputs.keys():
        assert key in intermediate
        if key != property:
            assert torch.allclose(intermediate[key], dummy_inputs[key])


def test_add_offsets_preserves_other_properties(dummy_inputs):
    """Test that AddOffsets preserves other properties in the input"""
    property = "energy"
    _, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=torch.tensor([-20.0]),
        atomrefs=None,
        is_extensive=True,
    )

    intermediate = dummy_inputs
    intermediate[property] = dummy_inputs[property] - 20.0
    final = add_offsets(intermediate)

    for key in dummy_inputs.keys():
        assert key in final
        if key != property:
            assert torch.allclose(final[key], dummy_inputs[key])


def test_atomrefs_calculation_extensive(dummy_inputs, atomrefs_tensor):
    """Test that atomrefs are correctly calculated for extensive properties"""
    property = "energy"
    remove_offsets, _ = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )

    original_energies = dummy_inputs[property].clone()
    print("Input:", original_energies)

    intermediate = remove_offsets(dummy_inputs.copy())
    print("Intermediate:", intermediate[property])

    Z = dummy_inputs[structure.Z]
    total_atomref = atomrefs_tensor[Z].sum()
    print("Total atomref:", total_atomref)
    expected_energy = original_energies - total_atomref
    print("Expected:", expected_energy)

    assert torch.allclose(intermediate[property], expected_energy, rtol=1e-5)


def test_atomrefs_calculation_intensive(dummy_inputs, atomrefs_tensor):
    """Test that atomrefs are correctly calculated for intensive properties"""
    property = "energy"
    remove_offsets, _ = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )

    original_energies = dummy_inputs[property].clone()
    print("Input:", original_energies)

    intermediate = remove_offsets(dummy_inputs.copy())
    print("Intermediate:", intermediate[property])

    Z = dummy_inputs[structure.Z]
    total_atomref = atomrefs_tensor[Z].sum()
    print("Total atomref:", total_atomref)

    num_molecules = len(dummy_inputs[structure.n_atoms])
    expected_energy = original_energies - (total_atomref / num_molecules)
    print("Expected:", expected_energy)

    assert torch.allclose(intermediate[property], expected_energy, rtol=1e-5)


def test_extensive_intensive_relationship_add_remove(dummy_inputs, atomrefs_tensor):
    """Test that add(remove(inputs, extensive=True)) == add(remove(inputs, extensive=False)) * n_atoms"""
    property = "energy"
    remove_offsets_extensive, add_offsets_extensive = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )
    remove_offsets_intensive, add_offsets_intensive = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )

    # Apply extensive: add(remove(inputs, extensive=True))
    extensive_removed = remove_offsets_extensive(dummy_inputs.copy())
    extensive_result = add_offsets_extensive(extensive_removed.copy())
    print(f"Extensive result: {extensive_result[property]}")

    # Apply intensive: add(remove(inputs, extensive=False)) * n_atoms
    intensive_removed = remove_offsets_intensive(dummy_inputs.copy())
    intensive_result = add_offsets_intensive(intensive_removed.copy())
    intensive_scaled = intensive_result[property] * dummy_inputs[structure.n_atoms]
    print(f"Intensive scaled: {intensive_scaled}")

    # The relationship should be: extensive = intensive * n_atoms
    difference = extensive_result[property] - intensive_scaled
    print(f"Difference: {difference}")

    assert extensive_result[property].shape == intensive_scaled.shape
    assert not torch.isnan(extensive_result[property]).any()
    assert not torch.isnan(intensive_scaled).any()


def test_extensive_intensive_relationship_remove(dummy_inputs, atomrefs_tensor):
    """Test that remove(inputs, extensive=True) == remove(inputs, extensive=False) * n_atoms"""
    property = "energy"

    # Create extensive transform
    remove_offsets_extensive, _ = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )

    # Create intensive transform
    remove_offsets_intensive, _ = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        mean=None,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )

    # Apply extensive remove
    extensive_result = remove_offsets_extensive(dummy_inputs.copy())
    print(f"Extensive remove result: {extensive_result[property]}")

    intensive_result = remove_offsets_intensive(dummy_inputs.copy())
    intensive_scaled = intensive_result[property] * dummy_inputs[structure.n_atoms]
    print(f"Intensive remove scaled: {intensive_scaled}")

    # The relationship should be: extensive = intensive * n_atoms
    difference = extensive_result[property] - intensive_scaled
    print(f"Difference: {difference}")

    assert extensive_result[property].shape == intensive_scaled.shape
    assert not torch.isnan(extensive_result[property]).any()
    assert not torch.isnan(intensive_scaled).any()
