import csv
import logging
import os
import shutil
import tempfile
from io import StringIO

import ase.io
import numpy as np
from tqdm import tqdm

from schnetpack.data.atoms import DownloadableASEAtomsData
from schnetpack.datasets.qm7x import download_and_check
from schnetpack.transform.base import Transform

__all__ = ["MP20"]


class MP20(DownloadableASEAtomsData):
    """
    MP-20: 45,229 inorganic crystals from the Materials Project with at most 20 atoms
    per cell. It is the benchmark of crystal generation introduced with CDVAE
    [#mp20_1]_ and used by DiffCSP [#mp20_2]_, FlowMM and MatterGen.

    The CSV files are read from the CDVAE repository at a fixed commit. Each crystal
    is read from its CIF as written there; pymatgen, which the reference code uses,
    rounds fractional coordinates within 1e-4 of a simple fraction (0.66657 to 2/3),
    so 647 crystals differ from the reference by at most 1e-4 in fractional
    coordinates.

    Each system carries its Materials Project id (`material_id`) as metadata. The
    published split (27,136 train, 9,047 val, 9,046 test) is stored in the metadata
    under ``splits``; select it with
    ``SubsamplePartitions(["train", "val", "test"], split_id=0)`` (data config
    ``data=mp20``).

    References:

    .. [#mp20_1] Xie, Fu, Ganea, Barzilay, Jaakkola. Crystal Diffusion Variational
       Autoencoder for Periodic Material Generation. ICLR (2022).
       https://github.com/txie-93/cdvae
    .. [#mp20_2] Jiao, Huang, Lin, Han, Chen, Lu, Liu. Crystal Structure Prediction
       by Joint Equivariant Diffusion. NeurIPS (2023).
    """

    # properties
    formation_energy_per_atom = "formation_energy_per_atom"
    band_gap = "band_gap"
    e_above_hull = "e_above_hull"

    # metadata
    MaterialId = "material_id"

    url = (
        "https://raw.githubusercontent.com/txie-93/cdvae/"
        "f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/data/mp_20/{split}.csv"
    )
    md5 = {
        "train": "1607b7a57559965c4fbf55f1df2cf46b",
        "val": "a7753df017ea03e6ce2efd1126ce973c",
        "test": "66e4b85c161c08cc056a5eaa7a995f7c",
    }

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
            raw_data_path: directory for the downloaded CSV files, which are kept
                there and reused if their checksums match. If None, they are
                downloaded to a temporary directory and deleted after parsing.
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
            MP20.formation_energy_per_atom: "eV",
            MP20.band_gap: "eV",
            MP20.e_above_hull: "eV",
        }

    def download(self) -> None:
        """
        Download the train, val and test CSV files and write their crystals to the
        ASE DB, in this order.
        """
        raw_dir = self.raw_data_path or tempfile.mkdtemp("mp20")
        try:
            os.makedirs(raw_dir, exist_ok=True)
            splits, start = {}, 0
            for split, md5 in self.md5.items():
                path = os.path.join(raw_dir, f"{split}.csv")
                download_and_check(self.url.format(split=split), path, md5)
                n = self._parse_csv(path)
                splits[split] = [list(range(start, start + n))]
                start += n
            self.update_metadata(splits=splits)
        finally:
            if self.raw_data_path is None:
                shutil.rmtree(raw_dir, ignore_errors=True)

    def _parse_csv(self, path: str) -> int:
        csv.field_size_limit(2**31 - 1)  # a CIF is one field
        atoms_list, property_list, atoms_metadata_list = [], [], []
        with open(path, newline="") as f:
            for row in tqdm(csv.DictReader(f), desc=os.path.basename(path)):
                atoms_list.append(ase.io.read(StringIO(row["cif"]), format="cif"))
                property_list.append(
                    {
                        key: np.array([float(row[key])], dtype=np.float64)
                        for key in self._native_property_units()
                    }
                )
                atoms_metadata_list.append({MP20.MaterialId: row["material_id"]})
        logging.info(f"Write {len(atoms_list)} crystals of {path} to the db ...")
        self.add_systems(property_list, atoms_list, atoms_metadata_list)
        return len(atoms_list)
