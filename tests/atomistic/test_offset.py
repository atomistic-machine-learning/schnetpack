import torch
import schnetpack.properties as structure
from schnetpack.transform import AddOffsets, RemoveOffsets


def get_dummy_inputs():
    # Create a simple molecule-like system (e.g. H2O)
    n_atoms = 3  # H, O, H

    inputs = {
        # Atomic numbers for H, O, H
        structure.Z: torch.tensor([1, 8, 1]),
        # Number of atoms
        structure.n_atoms: torch.tensor([n_atoms]),
        structure.idx_m: torch.tensor([0, 0, 0]),
        "energy": torch.tensor([-10.0]),
    }
    return inputs


def get_dummy_offsets():
    atomrefs = {
        "energy": [
            0.0,  # Z=0 (padding)
            -0.5,  # H (Z=1)
            0.0,  # He (Z=2)
            0.0,  # Li (Z=3)
            0.0,  # Be (Z=4)
            0.0,  # B (Z=5)
            0.0,  # C (Z=6)
            0.0,  # N (Z=7)
            -1.5,  # O (Z=8)
        ]
    }

    # Create mean energy value
    mean_energy = torch.tensor([-2.0])

    return {"mean_energy": mean_energy, "atomrefs": atomrefs}


def get_offset_transforms(use_mean=True, use_atomrefs=False):
    offsets = get_dummy_offsets()

    # Convert atomrefs to tensor only if we're using them
    atomrefs_tensor = None
    if use_atomrefs:
        atomrefs_tensor = torch.tensor(
            offsets["atomrefs"]["energy"], dtype=torch.float32
        )

    remove_offsets = RemoveOffsets(
        property="energy",
        remove_mean=use_mean,
        remove_atomrefs=use_atomrefs,
        atomrefs=atomrefs_tensor if use_atomrefs else None,
        property_mean=offsets["mean_energy"] if use_mean else None,
    )

    add_offsets = AddOffsets(
        property="energy",
        add_mean=use_mean,
        add_atomrefs=use_atomrefs,
        atomrefs=atomrefs_tensor if use_atomrefs else None,
        property_mean=offsets["mean_energy"] if use_mean else None,
    )

    return remove_offsets, add_offsets


inputs = get_dummy_inputs()
remove_offsets, add_offsets = get_offset_transforms()

intermediate = remove_offsets(inputs.copy())
final = add_offsets(intermediate)
is_equal = torch.allclose(final["energy"], inputs["energy"])

#### Test 1(use_mean=True, use_atomrefs=False)
print("\nOriginal energy:", inputs["energy"])
print("After remove_offsets:", intermediate["energy"])
print("After add_offsets:", final["energy"])
print("\nTransforms are inverse operations:", is_equal)

#### Test 2(use_mean=False, use_atomrefs=True)
remove_offsets, add_offsets = get_offset_transforms(use_mean=False, use_atomrefs=True)

intermediate = remove_offsets(inputs.copy())
final = add_offsets(intermediate)
is_equal = torch.allclose(final["energy"], inputs["energy"])

print("\nOriginal energy:", inputs["energy"])
print("After remove_offsets:", intermediate["energy"])
print("After add_offsets:", final["energy"])
print("\nTransforms are inverse operations:", is_equal)

#### Test 3(use_mean=True, use_atomrefs=True)
# remove_offsets, add_offsets = get_offset_transforms(use_mean=True, use_atomrefs=True)

# intermediate = remove_offsets(inputs.copy())
# final = add_offsets(intermediate)
# is_equal = torch.allclose(final['energy'], inputs['energy'])

# print("\nOriginal energy:", inputs['energy'])
# print("After remove_offsets:", intermediate['energy'])
# print("After add_offsets:", final['energy'])
# print("\nTransforms are inverse operations:", is_equal)
