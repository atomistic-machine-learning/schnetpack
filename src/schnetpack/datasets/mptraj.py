import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.units import GPa
from tqdm import tqdm

from schnetpack.data.atoms import AtomsDataError, DownloadableASEAtomsData
from schnetpack.datasets.qm7x import download_and_check
from schnetpack.transform.base import Transform

__all__ = ["MPTraj"]


def iter_json_object(path: str, chunk_size: int = 2**26) -> Iterator[tuple[str, Any]]:
    """
    Iterate over the (key, value) pairs of the JSON object stored in a file.

    Only one value is decoded at a time, from a buffer that is refilled from the
    file as needed. Memory is therefore bounded by the largest value, not by the
    file size (the MPtrj JSON has 12 GB).

    Args:
        path: path to a file holding one JSON object.
        chunk_size: number of characters read from the file per refill.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf, pos = "", 0

        def skip(chars: str) -> str:
            # advance past `chars` and return the next character
            nonlocal buf, pos
            while True:
                while pos < len(buf) and buf[pos] in chars:
                    pos += 1
                if pos < len(buf):
                    return buf[pos]
                buf, pos = f.read(chunk_size), 0
                if not buf:
                    raise AtomsDataError(f"Unexpected end of JSON file {path}.")

        def decode() -> Any:
            # decode the next value, reading more of the file while it is cut off
            nonlocal buf, pos
            while True:
                try:
                    value, pos = decoder.raw_decode(buf, pos)
                    return value
                except json.JSONDecodeError:
                    more = f.read(chunk_size)
                    if not more:
                        raise
                    buf, pos = buf[pos:] + more, 0

        whitespace = " \t\n\r"
        if skip(whitespace) != "{":
            raise AtomsDataError(f"{path} does not hold a JSON object.")
        pos += 1
        while skip(whitespace + ",") != "}":
            key = decode()
            skip(whitespace + ":")
            yield key, decode()


def structure_dict_to_atoms(structure: dict[str, Any]) -> Atoms:
    """
    Convert an ordered crystal in the dict format of `pymatgen.core.Structure`
    (``Structure.as_dict()``) to ASE atoms, without depending on pymatgen.

    Args:
        structure: dict with the lattice matrix under ``lattice.matrix`` and one
            entry per site with its species and fractional coordinates ``abc``.
    """
    numbers = []
    scaled_positions = []
    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1 or species[0]["occu"] != 1:
            raise AtomsDataError(f"Disordered site {site} is not supported.")
        numbers.append(atomic_numbers[species[0]["element"]])
        scaled_positions.append(site["abc"])

    return Atoms(
        numbers=numbers,
        scaled_positions=scaled_positions,
        cell=structure["lattice"]["matrix"],
        pbc=True,
    )


class MPTraj(DownloadableASEAtomsData):
    """
    Materials Project Trajectory (MPtrj) dataset of ~1.58 M DFT relaxation frames
    of ~146 k bulk crystals, with energies, forces, stresses and magnetic moments.
    The frames are taken from the GGA/GGA+U relaxations of the September 2022
    Materials Project. The dataset was created to train CHGNet [#mptrj_2]_.

    The raw data is provided by [#mptrj_1]_. By using it, you agree to the
    Materials Project terms of use (https://next-gen.materialsproject.org/about/terms).

    Conventions of the stored properties:

    - `energy` is the total energy after the MP2020 compatibility corrections,
      `uncorrected_energy` the raw VASP total energy. CHGNet was trained on the
      corrected energy per atom.
    - `stress` is converted from the raw VASP stress (kbar, positive under
      compression) to eV/Å³ with the ASE sign convention (positive under
      tension), the one predicted by `schnetpack.model.Forces(calc_stress=True)`.
      In GPa this is the CHGNet stress.
    - `magmoms` are the per-atom magnetic moments. They are missing for some
      frames; those frames store NaN for every atom (not 0).

    Each system carries its Materials Project id (`material_id`) and its frame
    id (`frame_id`, "task_id-calc_id-ionic_step") as metadata. The frames of
    one relaxation are strongly correlated, so split by material:
    `schnetpack.data.GroupSplit(splitting_key="material_idx")`.

    References:

    .. [#mptrj_1] `MPtrj dataset on figshare
       <https://doi.org/10.6084/m9.figshare.23713842.v2>`_
    .. [#mptrj_2] Deng et al. CHGNet as a pretrained universal neural network
       potential for charge-informed atomistic modelling.
       Nature Machine Intelligence 5, 1031 (2023).
    """

    # properties
    energy = "energy"
    uncorrected_energy = "uncorrected_energy"
    forces = "forces"
    stress = "stress"
    magmoms = "magmoms"

    # metadata
    MaterialId = "material_id"
    FrameId = "frame_id"
    # key of the integer material index in metadata["groups_ids"], for GroupSplit
    MaterialIdx = "material_idx"

    url = "https://ndownloader.figshare.com/files/41619375"
    json_file = "MPtrj_2022.9_full.json"
    json_md5 = "50ead5f27f9a4f6beb7564c4188f1e9f"

    def __init__(
        self,
        datapath: str,
        raw_data_path: str | None = None,
        load_properties: list[str] | None = None,
        transforms: list[Transform] | None = None,
        train_transforms: list[Transform] | None = None,
        val_transforms: list[Transform] | None = None,
        test_transforms: list[Transform] | None = None,
        subset_idx: list[int] | None = None,
        property_units: dict[str, str] | None = None,
        distance_unit: str | None = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            raw_data_path: directory for the downloaded MPtrj JSON (12 GB), which
                is kept there and reused if its checksum matches. If None, it is
                downloaded to a temporary directory next to `datapath` and
                deleted after parsing.
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """
        self.raw_data_path = raw_data_path

        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

    @staticmethod
    def _native_property_units() -> dict[str, str]:
        return {
            MPTraj.energy: "eV",
            MPTraj.uncorrected_energy: "eV",
            MPTraj.forces: "eV/Ang",
            MPTraj.stress: "eV/Ang/Ang/Ang",
            MPTraj.magmoms: "None",
        }

    def download(self) -> None:
        """
        Download the MPtrj JSON and write its frames to the ASE DB.
        """
        # The JSON has 12 GB, too large for /tmp on many machines.
        raw_dir = self.raw_data_path or tempfile.mkdtemp(
            "mptraj", dir=os.path.dirname(os.path.abspath(self.datapath))
        )
        try:
            os.makedirs(raw_dir, exist_ok=True)
            json_path = os.path.join(raw_dir, self.json_file)
            download_and_check(self.url, json_path, self.json_md5)
            self._parse_data(json_path)
        finally:
            if self.raw_data_path is None:
                shutil.rmtree(raw_dir, ignore_errors=True)

    def _parse_data(self, json_path: str, write_every: int = 10000) -> None:
        """
        Parse the MPtrj JSON (material id -> frame id -> frame) and add the frames
        to the dataset.
        """
        atoms_list = []
        property_list = []
        atoms_metadata_list = []
        material_idx = []

        logging.info("Parsing MPtrj and writing it to the db ...")
        materials = tqdm(iter_json_object(json_path), desc="MPtrj", unit=" materials")
        for idx, (material_id, frames) in enumerate(materials):
            for frame_id, frame in frames.items():
                atoms = structure_dict_to_atoms(frame["structure"])
                magmoms = frame.get("magmom")
                if magmoms is None:
                    magmoms = np.full(len(atoms), np.nan)

                atoms_list.append(atoms)
                property_list.append(
                    {
                        MPTraj.energy: np.array(
                            [frame["corrected_total_energy"]], dtype=np.float64
                        ),
                        MPTraj.uncorrected_energy: np.array(
                            [frame["uncorrected_total_energy"]], dtype=np.float64
                        ),
                        MPTraj.forces: np.array(frame["force"], dtype=np.float64),
                        # VASP kbar -> eV/Ang^3, and VASP's sign -> ASE's sign
                        MPTraj.stress: -0.1
                        * GPa
                        * np.array(frame["stress"], dtype=np.float64).reshape(1, 3, 3),
                        MPTraj.magmoms: np.array(magmoms, dtype=np.float64),
                    }
                )
                atoms_metadata_list.append(
                    {MPTraj.MaterialId: material_id, MPTraj.FrameId: frame_id}
                )
                material_idx.append(idx)

            if len(atoms_list) >= write_every:
                self.add_systems(property_list, atoms_list, atoms_metadata_list)
                atoms_list, property_list, atoms_metadata_list = [], [], []

        if atoms_list:
            self.add_systems(property_list, atoms_list, atoms_metadata_list)

        self.update_metadata(groups_ids={MPTraj.MaterialIdx: material_idx})
        logging.info("Done.")
