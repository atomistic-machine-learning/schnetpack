import os
import numpy as np


def train_test_split(
    data,
    num_train=None,
    num_val=None,
    split_file=None,
    stratify_partitions=False,
    num_per_partition=False,
):
    """
    Splits the dataset into train/validation/test splits, writes split to
    an npz file and returns subsets. Either the sizes of training and
    validation split or an existing split file with split indices have to
    be supplied. The remaining data will be used in the test dataset.

    Args:
        num_train (int): number of training examples
        num_val (int): number of validation examples
        split_file (str): Path to split file. If file exists, splits will
                          be loaded. Otherwise, a new file will be created
                          where the generated split is stored.

    Returns:
        schnetpack.data.AtomsData: training dataset
        schnetpack.data.AtomsData: validation dataset
        schnetpack.data.AtomsData: test dataset

    """
    if split_file is not None and os.path.exists(split_file):
        S = np.load(split_file)
        train_idx = S["train_idx"].tolist()
        val_idx = S["val_idx"].tolist()
        test_idx = S["test_idx"].tolist()
    else:
        if num_train is None or num_val is None:
            raise ValueError(
                "You have to supply either split sizes (num_train /"
                + " num_val) or an npz file with splits."
            )

        assert num_train + num_val <= len(
            data
        ), "Dataset is smaller than num_train + num_val!"

        num_train = num_train if num_train > 1 else num_train * len(data)
        num_val = num_val if num_val > 1 else num_val * len(data)
        num_train = int(num_train)
        num_val = int(num_val)

        if stratify_partitions:
            partitions = data.get_metadata("partitions")
            n_partitions = len(partitions)
            if num_per_partition:
                num_train_part = num_train
                num_val_part = num_val
            else:
                num_train_part = num_train // n_partitions
                num_val_part = num_val // n_partitions

            train_idx = []
            val_idx = []
            test_idx = []
            for start, stop in partitions.values():
                idx = np.random.permutation(np.arange(start, stop))
                train_idx += idx[:num_train_part].tolist()
                val_idx += idx[num_train_part : num_train_part + num_val_part].tolist()
                test_idx += idx[num_train_part + num_val_part :].tolist()

        else:
            idx = np.random.permutation(len(data))
            train_idx = idx[:num_train].tolist()
            val_idx = idx[num_train : num_train + num_val].tolist()
            test_idx = idx[num_train + num_val :].tolist()

        if split_file is not None:
            np.savez(
                split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )

    train = data.create_subset(train_idx)
    val = data.create_subset(val_idx)
    test = data.create_subset(test_idx)
    return train, val, test
