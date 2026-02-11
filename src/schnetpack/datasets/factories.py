from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

from schnetpack.data import AtomsDataFormat, load_dataset

from schnetpack.datasets.md17 import MD17
from schnetpack.datasets.rmd17 import rMD17

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "get_dataset_id",
    "build_dataset_by_name",
    "component_db_path",
]

# ---------------------------
# Registry
# ---------------------------


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: int
    builder_name: str  # name used in the factory dispatch below


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "MD17": DatasetSpec(dataset_id=0, builder_name="MD17"),
    "rMD17": DatasetSpec(dataset_id=1, builder_name="rMD17"),
}


def get_dataset_id(dataset_name: str) -> int:
    """
    Return the stable integer ID for a dataset name.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{dataset_name}' not registered. Add it to DATASET_REGISTRY."
        )
    return DATASET_REGISTRY[dataset_name].dataset_id


# ---------------------------
# Factory (build/load datasets)
# ---------------------------


def component_db_path(dataset_root: str, dataset_name: str, molecule: str) -> str:
    """
    Standard path for component datasets:
      <dataset_root>/<dataset_name>/<molecule>.db
    """
    folder = os.path.join(dataset_root, dataset_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{molecule}.db")


def build_dataset_by_name(
    name: str,
    molecule: str,
    dataset_root: str,
    load_properties: Optional[List[str]] = None,
) -> Any:
    """
    Ensure the component dataset exists on disk and return the loaded ASEAtomsData.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset name: {name}. Add it to DATASET_REGISTRY.")

    spec = DATASET_REGISTRY[name]
    dbpath = component_db_path(dataset_root, name, molecule)

    # If DB not there, create it using the matching DataModule
    if not os.path.exists(dbpath):
        if spec.builder_name == "MD17":
            dm = MD17(datapath=dbpath, molecule=molecule, batch_size=1)
            dm.prepare_data()

        elif spec.builder_name == "rMD17":
            dm = rMD17(datapath=dbpath, molecule=molecule, batch_size=1)
            dm.prepare_data()

        else:
            raise NotImplementedError(
                f"Builder '{spec.builder_name}' not implemented in factories.py"
            )

    # Load into ASEAtomsData
    ds = load_dataset(
        dbpath,
        format=AtomsDataFormat.ASE,
        load_properties=load_properties,
    )
    return ds
