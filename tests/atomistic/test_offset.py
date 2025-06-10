import pytest
import torch
import schnetpack.properties as structure
from schnetpack.transform import AddOffsets, RemoveOffsets


# === Fixtures ===
@pytest.fixture
def dummy_inputs_extensive():
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
def dummy_inputs_intensive():
    """
    Same structure as extensive but with intensive (per-atom) properties
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

    # For intensive properties, energy per atom
    n_atoms = torch.tensor([len(ethanol_Z), len(methanol_Z), len(water_Z)])
    energies_per_atom = torch.tensor([-170.0 / 9, -115.0 / 6, -76.0 / 3])  # Per atom

    inputs = {
        structure.Z: torch.tensor(all_Z),
        structure.n_atoms: n_atoms,
        structure.idx_m: torch.tensor(all_idx),
        "energy": energies_per_atom,
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
def test_remove_add_offsets_mean_none_extensive(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_mean_extensive(dummy_inputs_intensive, mean_tensor):
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

    intermediate = remove_offsets(dummy_inputs_intensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_intensive[property])


def test_remove_add_offsets_mean_none_intensive(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_mean_intensive(dummy_inputs_intensive, mean_tensor):
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

    intermediate = remove_offsets(dummy_inputs_intensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_intensive[property])


def test_remove_add_offsets_atomrefs_none_extensive(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_atomrefs_tensor_extensive(
    dummy_inputs_intensive, atomrefs_tensor
):
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

    intermediate = remove_offsets(dummy_inputs_intensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_intensive[property])


def test_remove_add_offsets_atomrefs_none(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_atomrefs_tensor(dummy_inputs_intensive, atomrefs_tensor):
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

    intermediate = remove_offsets(dummy_inputs_intensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_intensive[property])


def test_remove_add_offsets_both_mean_and_atomrefs_none_extensive(
    dummy_inputs_extensive,
):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_both_mean_and_atomrefs_extensive(
    dummy_inputs_extensive, mean_tensor, atomrefs_tensor
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_both_mean_and_atomrefs_none(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_both_mean_and_atomrefs_extensive(
    dummy_inputs_extensive, mean_tensor, atomrefs_tensor
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_no_transforms_extensive(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_extensive[property])


def test_remove_add_offsets_no_transforms_intensive(dummy_inputs_intensive):
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

    intermediate = remove_offsets(dummy_inputs_intensive.copy())
    final = add_offsets(intermediate.copy())

    # Check that transforms are inverse operations
    assert torch.allclose(final[property], dummy_inputs_intensive[property])


def test_remove_offsets_preserves_other_properties(dummy_inputs_extensive):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())

    # Check that all original properties are preserved
    for key in dummy_inputs_extensive.keys():
        assert key in intermediate
        if key != property:  # energy property should be transformed
            assert torch.allclose(intermediate[key], dummy_inputs_extensive[key])


def test_add_offsets_preserves_other_properties(dummy_inputs_extensive):
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

    # Create intermediate data (simulating after remove_offsets)
    intermediate = dummy_inputs_extensive.copy()
    intermediate[property] = (
        dummy_inputs_extensive[property] - 20.0
    )  # Simulate removed mean

    final = add_offsets(intermediate.copy())

    # Check that all original properties are preserved
    for key in dummy_inputs_extensive.keys():
        assert key in final
        if key != property:  # energy property should be transformed
            assert torch.allclose(final[key], dummy_inputs_extensive[key])


def test_atomrefs_calculation_extensive(dummy_inputs_extensive, atomrefs_tensor):
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

    intermediate = remove_offsets(dummy_inputs_extensive.copy())

    # For extensive properties, atomrefs should be summed per molecule
    # Ethanol: 9 atoms (1C + 3H + 1C + 2H + 1O + 1H) = -1.0 + 3*(-0.5) + -1.0 + 2*(-0.5) + -1.5 + (-0.5) = -6.5
    # Methanol: 6 atoms (1C + 3H + 1O + 1H) = -1.0 + 3*(-0.5) + -1.5 + (-0.5) = -4.5
    # Water: 3 atoms (1O + 2H) = -1.5 + 2*(-0.5) = -2.5

    expected_removed = torch.tensor([-6.5, -4.5, -2.5])
    expected_energy = dummy_inputs_extensive[property] - expected_removed

    assert torch.allclose(intermediate[property], expected_energy, rtol=1e-5)


def test_atomrefs_calculation_intensive(dummy_inputs_intensive, atomrefs_tensor):
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

    intermediate = remove_offsets(dummy_inputs_intensive.copy())

    # For intensive properties, atomrefs should be averaged per molecule
    # Ethanol: 9 atoms, average = -6.5/9 = -0.722...
    # Methanol: 6 atoms, average = -4.5/6 = -0.75
    # Water: 3 atoms, average = -2.5/3 = -0.833...

    expected_removed = torch.tensor([-6.5 / 9, -4.5 / 6, -2.5 / 3])
    expected_energy = dummy_inputs_intensive[property] - expected_removed

    assert torch.allclose(intermediate[property], expected_energy, rtol=1e-5)
