import pytest
import torch
import schnetpack.properties as structure
from schnetpack.transform import AddOffsets, RemoveOffsets
import copy


# === Fixtures ===
@pytest.fixture
def dummy_inputs():
    """
    Structure:
    - C2H: 2C, 1H (3 atoms)
    - C4H2: 4C, 2H (6 atoms)
    - CH4: 1C, 4H (5 atoms)
    """
    c2h_Z = [6, 6, 1]  # 2C, 1H
    c4h2_Z = [6, 6, 6, 6, 1, 1]  # 4C, 2H
    ch4_Z = [6, 1, 1, 1, 1]  # 1C, 4H

    all_Z = c2h_Z + c4h2_Z + ch4_Z

    c2h_idx = [0] * len(c2h_Z)
    c4h2_idx = [1] * len(c4h2_Z)
    ch4_idx = [2] * len(ch4_Z)
    all_idx = c2h_idx + c4h2_idx + ch4_idx

    n_atoms = torch.tensor([len(c2h_Z), len(c4h2_Z), len(ch4_Z)])
    energies = torch.tensor([83.0, 166.0, 50.0])

    inputs = {
        structure.Z: torch.tensor(all_Z),
        structure.n_atoms: n_atoms,
        structure.idx_m: torch.tensor(all_idx),
        "energy": energies,
        "energy_per_atom": energies / n_atoms,
    }

    return inputs


@pytest.fixture
def atomrefs_tensor():
    """Create atomrefs tensor with H=2eV, C=40eV, all others 0 (zmax=100)"""
    atomrefs_values = [0.0] * 100
    atomrefs_values[1] = 2.0  # H (Z=1)
    atomrefs_values[6] = 40.0  # C (Z=6)
    return torch.tensor(atomrefs_values, dtype=torch.float32)


# === Helper Functions ===
def split_batch_into_molecules(inputs):
    """
    Split a batch of molecules into individual molecule dictionaries.

    Args:
        inputs: Dictionary containing batch data with idx_m for molecule indices

    Returns:
        List of dictionaries, each containing data for one molecule
    """
    Z = inputs[structure.Z]
    idx_m = inputs[structure.idx_m]
    n_atoms = inputs[structure.n_atoms]
    energies = inputs["energy"]
    energy_per_atom = inputs["energy_per_atom"]

    molecules = []
    for mol_idx in range(len(n_atoms)):
        # Get atoms for this molecule
        mol_mask = idx_m == mol_idx
        mol_Z = Z[mol_mask]

        # Create single molecule input
        mol_input = {
            structure.Z: mol_Z,
            structure.n_atoms: torch.tensor([n_atoms[mol_idx]]),
            structure.idx_m: torch.zeros(
                len(mol_Z), dtype=torch.long
            ),  # All atoms belong to molecule 0
            "energy": torch.tensor([energies[mol_idx]]),
            "energy_per_atom": torch.tensor([energy_per_atom[mol_idx]]),
        }
        molecules.append(mol_input)

    return molecules


def combine_molecules_into_batch(molecules):
    """
    Combine individual molecule dictionaries back into a batch.

    Args:
        molecules: List of molecule dictionaries

    Returns:
        Dictionary containing batch data
    """
    all_Z = []
    all_idx_m = []
    all_n_atoms = []
    all_energies = []
    all_energy_per_atom = []

    for mol_idx, mol in enumerate(molecules):
        all_Z.extend(mol[structure.Z].tolist())
        all_idx_m.extend([mol_idx] * len(mol[structure.Z]))
        all_n_atoms.append(mol[structure.n_atoms].item())
        all_energies.append(mol["energy"].item())
        all_energy_per_atom.append(mol["energy_per_atom"].item())

    return {
        structure.Z: torch.tensor(all_Z),
        structure.n_atoms: torch.tensor(all_n_atoms),
        structure.idx_m: torch.tensor(all_idx_m),
        "energy": torch.tensor(all_energies),
        "energy_per_atom": torch.tensor(all_energy_per_atom),
    }


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
def test_extensive_mean_and_atomref(dummy_inputs, atomrefs_tensor):
    """Extensive: use both mean and atomref"""
    property = "energy"
    mean_tensor = torch.tensor([0.355])
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=mean_tensor,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )

    print("\n=== Processing molecules individually ===")
    print("Original energies:", dummy_inputs["energy"])

    molecules = split_batch_into_molecules(dummy_inputs)

    processed_molecules = []
    for _, mol in enumerate(molecules):
        processed_mol = remove_offsets(mol.copy())
        processed_molecules.append(processed_mol)

        print(f"  Energy after RemoveOffsets: {processed_mol[property]}")

    intermediate_batch = combine_molecules_into_batch(processed_molecules)

    expected_individual = [-0.065, -0.13, 0.225]
    expected = torch.tensor(expected_individual)
    print(f"\nExpected energies after RemoveOffsets: {expected}")
    print(f"Actual energies after RemoveOffsets: {intermediate_batch[property]}")

    assert torch.allclose(intermediate_batch[property], expected, atol=1e-1)

    ## Add offsets: batch
    final_batch = add_offsets(intermediate_batch.copy())
    print(f"Final energies after AddOffsets (batch): {final_batch[property]}")
    assert torch.allclose(final_batch[property], dummy_inputs[property], atol=1e-1)

    ## Add offsets: single molecule
    restored_molecules = []
    for i, mol in enumerate(processed_molecules):
        restored = add_offsets(mol.copy())
        restored_molecules.append(restored)

    restored_batch = combine_molecules_into_batch(restored_molecules)
    print(
        f"\nFinal energies after AddOffsets (single molecule): {restored_batch[property]}"
    )
    print(f"Original energies: {dummy_inputs[property]}")
    assert torch.allclose(restored_batch[property], dummy_inputs[property], atol=1e-2)


def test_extensive_mean_only(dummy_inputs):
    """Extensive: use mean only"""
    property = "energy"
    mean_tensor = torch.tensor([21.77])
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=mean_tensor,
        is_extensive=True,
    )
    molecules = split_batch_into_molecules(dummy_inputs)

    processed_molecules = []
    for i, mol in enumerate(molecules):
        processed_mol = remove_offsets(mol.copy())
        processed_molecules.append(processed_mol)

    intermediate_batch = combine_molecules_into_batch(processed_molecules)

    expected = torch.tensor([17.69, 35.33, -58.88])
    print(f"\nExpected energies after RemoveOffsets: {expected}")
    print(f"Actual energies after RemoveOffsets: {intermediate_batch[property]}")

    assert torch.allclose(intermediate_batch[property], expected, atol=1e-1)

    ## Add offsets: batch
    final_batch = add_offsets(intermediate_batch.copy())
    print(f"Final energies after AddOffsets (batch): {final_batch[property]}")
    assert torch.allclose(final_batch[property], dummy_inputs[property], atol=1e-1)

    ## Add offsets: single molecule
    restored_molecules = []
    for i, mol in enumerate(processed_molecules):
        restored = add_offsets(mol.copy())
        restored_molecules.append(restored)

    restored_batch = combine_molecules_into_batch(restored_molecules)
    print(
        f"\nFinal energies after AddOffsets (single molecule): {restored_batch[property]}"
    )
    print(f"Original energies: {dummy_inputs[property]}")
    assert torch.allclose(restored_batch[property], dummy_inputs[property], atol=1e-2)


def test_extensive_atomref_only(dummy_inputs, atomrefs_tensor):
    """Extensive: use atomref only"""
    property = "energy"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        atomrefs=atomrefs_tensor,
        is_extensive=True,
    )
    molecules = split_batch_into_molecules(dummy_inputs)

    processed_molecules = []
    for i, mol in enumerate(molecules):
        processed_mol = remove_offsets(mol.copy())
        processed_molecules.append(processed_mol)

    intermediate_batch = combine_molecules_into_batch(processed_molecules)

    expected = torch.tensor([1.01, 2.02, 2.0])
    print(f"\nExpected energies after RemoveOffsets: {expected}")
    print(f"Actual energies after RemoveOffsets: {intermediate_batch['energy']}")

    assert torch.allclose(intermediate_batch[property], expected, atol=1e-1)

    ## Add offsets: batch
    final_batch = add_offsets(intermediate_batch.copy())
    print(f"Final energies after AddOffsets (batch): {final_batch[property]}")
    assert torch.allclose(final_batch[property], dummy_inputs[property], atol=1e-1)

    ## Add offsets: single molecule
    restored_molecules = []
    for i, mol in enumerate(processed_molecules):
        restored = add_offsets(mol.copy())
        restored_molecules.append(restored)

    restored_batch = combine_molecules_into_batch(restored_molecules)
    print(
        f"\nFinal energies after AddOffsets (single molecule): {restored_batch[property]}"
    )
    print(f"Original energies: {dummy_inputs[property]}")
    assert torch.allclose(restored_batch[property], dummy_inputs[property], atol=1e-2)


def test_intensive_mean_and_atomref(dummy_inputs, atomrefs_tensor):
    """Intensive: use both mean and atomref"""
    property = "energy_per_atom"
    mean_tensor = torch.tensor([0.355])
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=True,
        mean=mean_tensor,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )
    molecules = split_batch_into_molecules(dummy_inputs)

    processed_molecules = []
    for _, mol in enumerate(molecules):
        processed_mol = remove_offsets(mol.copy())
        print("\nprocessed molecule", processed_mol)
        processed_molecules.append(processed_mol)

    intermediate_batch = combine_molecules_into_batch(processed_molecules)

    expected = torch.tensor([-0.022, -0.022, 0.045])
    print(f"\nExpected energies after RemoveOffsets: {expected}")
    print(f"Actual energies after RemoveOffsets: {intermediate_batch[property]}")

    assert torch.allclose(intermediate_batch[property], expected, atol=1e-1)

    ## Add offsets: batch
    final_batch = add_offsets(intermediate_batch.copy())
    print(f"Final energies after AddOffsets (batch): {final_batch[property]}")
    assert torch.allclose(final_batch[property], dummy_inputs[property], atol=1e-3)

    ## Add offsets: single molecule
    restored_molecules = []
    for i, mol in enumerate(processed_molecules):
        restored = add_offsets(mol.copy())
        restored_molecules.append(restored)

    restored_batch = combine_molecules_into_batch(restored_molecules)
    print(
        f"\nFinal energies after AddOffsets (single molecule): {restored_batch[property]}"
    )
    print(f"Original energies: {dummy_inputs[property]}")
    assert torch.allclose(restored_batch[property], dummy_inputs[property], atol=1e-2)


def test_intensive_mean_only(dummy_inputs):
    """Intensive: use mean only"""
    property = "energy_per_atom"
    mean_tensor = torch.tensor([21.77])
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=True,
        use_atomrefs=False,
        mean=mean_tensor,
        is_extensive=False,
    )
    molecules = split_batch_into_molecules(dummy_inputs)

    processed_molecules = []
    for _, mol in enumerate(molecules):
        processed_mol = remove_offsets(mol.copy())
        processed_molecules.append(processed_mol)

    intermediate_batch = combine_molecules_into_batch(processed_molecules)

    expected = torch.tensor([5.89, 5.89, -11.77])
    print(f"\nExpected energies after RemoveOffsets: {expected}")
    print(f"Actual energies after RemoveOffsets: {intermediate_batch[property]}")

    assert torch.allclose(intermediate_batch[property], expected, atol=1e-1)

    ## Add offsets: batch
    final_batch = add_offsets(intermediate_batch.copy())
    print(f"Final energies after AddOffsets (batch): {final_batch[property]}")
    assert torch.allclose(final_batch[property], dummy_inputs[property], atol=1e-1)

    ## Add offsets: single molecule
    restored_molecules = []
    for i, mol in enumerate(processed_molecules):
        restored = add_offsets(mol.copy())
        restored_molecules.append(restored)

    restored_batch = combine_molecules_into_batch(restored_molecules)
    print(
        f"\nFinal energies after AddOffsets (single molecule): {restored_batch[property]}"
    )
    print(f"Original energies: {dummy_inputs[property]}")
    assert torch.allclose(restored_batch[property], dummy_inputs[property], atol=1e-2)


def test_intensive_atomref_only(dummy_inputs, atomrefs_tensor):
    """Intensive: use atomref only"""
    property = "energy_per_atom"
    remove_offsets, add_offsets = get_offset_transforms(
        property=property,
        use_mean=False,
        use_atomrefs=True,
        atomrefs=atomrefs_tensor,
        is_extensive=False,
    )
    molecules = split_batch_into_molecules(dummy_inputs.copy())

    processed_molecules = []
    for _, mol in enumerate(molecules):
        processed_mol = remove_offsets(mol.copy())
        processed_molecules.append(processed_mol)

    intermediate_batch = combine_molecules_into_batch(processed_molecules)

    expected = torch.tensor([0.33, 0.33, 0.4])
    print(f"\nExpected energies after RemoveOffsets: {expected}")
    print(f"Actual energies after RemoveOffsets: {intermediate_batch[property]}")

    assert torch.allclose(intermediate_batch[property], expected, atol=1e-1)

    ## Add offsets: batch
    final_batch = add_offsets(intermediate_batch.copy())
    print(f"Final energies after AddOffsets (batch): {final_batch[property]}")
    assert torch.allclose(final_batch[property], dummy_inputs[property], atol=1e-1)

    ## Add offsets: single molecule
    restored_molecules = []
    for i, mol in enumerate(processed_molecules):
        restored = add_offsets(mol.copy())
        restored_molecules.append(restored)

    restored_batch = combine_molecules_into_batch(restored_molecules)
    print(
        f"\nFinal energies after AddOffsets (single molecule): {restored_batch[property]}"
    )
    print(f"Original energies: {dummy_inputs[property]}")
    assert torch.allclose(restored_batch[property], dummy_inputs[property], atol=1e-2)
