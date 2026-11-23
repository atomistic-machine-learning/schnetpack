"""spk convert: update the metadata of an ASE DB dataset in place
(``configs/convert.yaml``)."""

import hydra
import numpy as np
from ase.db import connect
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../configs", config_name="convert", version_base="1.2")
def main(config: DictConfig):
    """
    Update the metadata of the ASE DB dataset ``db`` in place, e.g. to make a
    dataset created with SchNetPack 1.0 loadable by the current version.

    - Converts old-style atom references (array + `atref_labels`) to a dict
      mapping each property to its per-element references.
    - Sets the distance unit (``distunit``) and property units (``propunit``).
    - Optionally prepends a dimension to the given properties in every row
      (``expand_property_dims``), as needed for old FieldSchNet datasets.

    The database is modified in place, so back it up first.
    """
    with connect(config.db) as db:
        meta = db.metadata
    print(meta)

    if "atomrefs" not in meta.keys():
        meta["atomrefs"] = {}
    elif "atref_labels" in meta.keys():
        old_atref = np.array(meta["atomrefs"])
        new_atomrefs = {}
        labels = meta["atref_labels"]
        if type(labels) is str:
            labels = [labels]
        for i, label in enumerate(labels):
            print(i, label, old_atref[:, i])
            new_atomrefs[label] = list(old_atref[:, i])
        meta["atomrefs"] = new_atomrefs
        del meta["atref_labels"]

    if config.distunit:
        if config.distunit == "A":
            raise ValueError(
                "The provided unit (A for Ampere) is not a valid distance unit according to the ASE unit"
                " definitions. You probably mean `Ang`/`Angstrom`. Please also check your property units!"
            )
        meta["_distance_unit"] = config.distunit

    if config.propunit:
        meta.setdefault("_property_unit_dict", {}).update(dict(config.propunit))

    with connect(config.db) as db:
        db.metadata = meta

    if config.expand_property_dims:
        with connect(config.db) as db:
            for i in tqdm(range(len(db))):
                atoms_row = db.get(i + 1)
                data = {}
                for p, v in atoms_row.data.items():
                    if p in config.expand_property_dims:
                        data[p] = np.expand_dims(v, 0)
                    else:
                        data[p] = v
                db.update(i + 1, data=data)
