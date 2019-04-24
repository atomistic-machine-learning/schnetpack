import numpy as np
import os
import schnetpack as spk
from shutil import copyfile
from torch.utils.data.sampler import RandomSampler


def get_statistics(split_path, logging, train_loader, train_args, atomref):
    split_data = np.load(split_path)
    if "mean" in split_data.keys():
        mean = split_data["mean"].item()
        stddev = split_data["stddev"].item()
        calc_stats = False
        logging.info("cached statistics was loaded...")
    else:
        mean, stddev = train_loader.get_statistics(train_args.property, True, atomref)
        np.savez(
            split_path,
            train_idx=split_data["train_idx"],
            val_idx=split_data["val_idx"],
            test_idx=split_data["test_idx"],
            mean=mean,
            stddev=stddev,
        )
    return mean, stddev


def get_loaders(logging, args, dataset, split_path):
    if args.mode == "train":
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info("create splits...")
    data_train, data_val, data_test = dataset.create_splits(
        *args.split, split_file=split_path
    )

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
