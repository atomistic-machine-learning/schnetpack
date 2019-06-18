import numpy as np
import torch
import os
import schnetpack as spk
from shutil import copyfile
from torch.utils.data.sampler import RandomSampler
from scripts.script_utils.script_error import ScriptError


def get_statistics(
    split_path, train_loader, args, atomref, per_atom=False, logging=None
):
    """
    Get statistics for molecular properties. Use split file if possible.

    Args:
        split_path (str): path to the split file
        train_loader (spk.data.AtomsLoader): dataloader for training set
        args (argparse.Namespace): parsed script arguments
        atomref (dict): atomic references
        logging: logger

    Returns:
        mean (dict): mean values for the selected properties
        stddev (dict): stddev values for the selected properties
    """
    # check if split file exists
    if not os.path.exists(split_path):
        raise ScriptError("No split file found ad {}".format(split_path))
    split_data = np.load(split_path)

    # check if split file contains statistical data
    if "mean" in split_data.keys():
        mean = {args.property: torch.from_numpy(split_data["mean"])}
        stddev = {args.property: torch.from_numpy(split_data["stddev"])}
        if logging is not None:
            logging.info("cached statistics was loaded...")

    # calculate statistical data
    else:
        mean, stddev = train_loader.get_statistics(args.property, per_atom, atomref)
        np.savez(
            split_path,
            train_idx=split_data["train_idx"],
            val_idx=split_data["val_idx"],
            test_idx=split_data["test_idx"],
            mean=mean[args.property].numpy(),
            stddev=stddev[args.property].numpy(),
        )

    return mean, stddev


def get_loaders(args, dataset, split_path, logging=None):

    if logging is not None:
        logging.info("create splits...")

    data_train, data_val, data_test = dataset.create_splits(
        *args.split, split_file=split_path
    )
    if logging is not None:
        logging.info("load data...")

    train_loader = spk.data.AtomsLoader(
        data_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(data_train),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = spk.data.AtomsLoader(
        data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True
    )
    test_loader = spk.data.AtomsLoader(
        data_test, batch_size=args.batch_size, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader, test_loader
