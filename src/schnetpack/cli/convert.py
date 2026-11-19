"""spkconvert: update the metadata of an ASE DB dataset in place."""

import argparse

import numpy as np
from ase.db import connect
from tqdm import tqdm


def main():
    """
    Update the metadata of an ASE DB dataset in place, e.g. to make a dataset
    created with SchNetPack 1.0 loadable by the current version.

    - Converts old-style atom references (array + `atref_labels`) to a dict
      mapping each property to its per-element references.
    - Sets the distance unit (`--distunit`) and property units (`--propunit`).
    - Optionally prepends a dimension to the given properties in every row
      (`--expand_property_dims`), as needed for old FieldSchNet datasets.

    The database is modified in place, so back it up first.

    """
    parser = argparse.ArgumentParser(
        description="Set units of an ASE dataset, e.g. to convert from SchNetPack 1.0 to the new format."
    )
    parser.add_argument(
        "data_path",
        help="Path to ASE DB dataset",
    )
    parser.add_argument(
        "--distunit",
        help="Distance unit as string, corresponding to ASE units (e.g. `Ang`)",
    )
    parser.add_argument(
        "--propunit",
        help="Property units as string, corresponding "
        "to ASE units (e.g. `kcal/mol/Ang`), in the form: `property1:unit1,property2:unit2`",
    )
    parser.add_argument(
        "--expand_property_dims",
        default=[],
        nargs="+",
        help="Expanding the first dimension of the given property "
        "(required for example for old FieldSchNet datasets). "
        "Add property names here in the form 'property1 property2 property3'",
    )
    args = parser.parse_args()
    with connect(args.data_path) as db:
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

    if args.distunit:
        if args.distunit == "A":
            raise ValueError(
                "The provided unit (A for Ampere) is not a valid distance unit according to the ASE unit"
                " definitions. You probably mean `Ang`/`Angstrom`. Please also check your property units!"
            )
        meta["_distance_unit"] = args.distunit

    if args.propunit:
        if "_property_unit_dict" not in meta.keys():
            meta["_property_unit_dict"] = {}

        for p in args.propunit.split(","):
            prop, unit = p.split(":")
            meta["_property_unit_dict"][prop] = unit

    with connect(args.data_path) as db:
        db.metadata = meta

    if args.expand_property_dims is not None and len(args.expand_property_dims) > 0:
        with connect(args.data_path) as db:
            for i in tqdm(range(len(db))):
                atoms_row = db.get(i + 1)
                data = {}
                for p, v in atoms_row.data.items():
                    if p in args.expand_property_dims:
                        data[p] = np.expand_dims(v, 0)
                    else:
                        data[p] = v
                db.update(i + 1, data=data)


if __name__ == "__main__":
    main()
