import os
from ase.db import connect
from .atoms import AtomsData, AtomsDataError


__all__ = ["merge_datasets", "save_dataset"]


def merge_datasets(merged_dbpath, dbpaths, **mergedb_kwargs):

    if type(dbpaths) is dict:
        names = dbpaths.keys()
        dbpaths = dbpaths.values()
    else:
        names = [dbp.split("/")[-1].split(".")[0] for dbp in dbpaths]

    partitions = {}
    offset = 0

    partition_meta = {}
    with connect(merged_dbpath, use_lock_file=False) as dst:
        for name, dbp in zip(names, dbpaths):
            start = offset

            if name in partitions.keys():
                count = 1
                while name + "_" + str(count) in partitions.keys():
                    count += 1
                name = name + "_" + str(count)

            with connect(dbp) as src:
                length = src.count()
                end = offset + length
                partition_meta[name] = src.metadata

                for row in src.select():
                    at = row.toatoms()
                    dst.write(at, key_value_pairs=row.key_value_pairs, data=row.data)
            partitions[name] = (start, end)
            offset += length

    metadata = {"partition_meta": partition_meta, "partitions": partitions}
    dst.metadata = metadata

    return AtomsData(merged_dbpath, **mergedb_kwargs)


def save_dataset(dbpath, dataset, overwrite=False):
    """
    Write dataset instance to ase-db file.

    Args:
        dbpath (str): path to the new database
        dataset (spk.data.ConcatAtomsData or spk.data.AtomsDataSubset): dataset
            instance to be stored
        overwrite (bool): overwrite existing database

    """
    # check if path exists
    if os.path.exists(dbpath):
        if overwrite:
            os.remove(dbpath)
        raise AtomsDataError(
            "The selected dbpath does already exist. Set overwrite=True or change "
            "dbpath."
        )

    # build metadata
    metadata = dict()
    metadata["atomref"] = dataset.atomref
    metadata["available_properties"] = dataset.available_properties

    # write dataset
    with connect(dbpath) as conn:
        # write metadata
        conn.metadata = metadata
        # write datapoints
        for idx in range(len(dataset)):
            atms, data = dataset.get_properties(idx)
            # filter available properties
            data_clean = dict()
            for pname, prop in data.items():
                if pname.startswith("_"):
                    continue
                if pname in dataset.available_properties:
                    data_clean[pname] = prop

            conn.write(atms, data=data_clean)
