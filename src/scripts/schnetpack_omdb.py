#!/usr/bin/env python
import logging
import os
from shutil import copyfile


import numpy as np
import schnetpack.output_modules
import torch
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
from schnetpack.datasets import OrganicMaterialsDatabase
from scripts.script_utils import setup_run, get_model, get_representation, train,\
    evaluate, get_main_parser, add_subparsers

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def add_omdb_arguments(parser):
    parser.add_argument(
        "--property",
        type=str,
        help="Organic Materials Database property to be predicted (default: %(default)s)",
        default="band_gap",
        choices=OrganicMaterialsDatabase.properties,
    )


if __name__ == "__main__":

    parser = get_main_parser()
    add_omdb_arguments(parser)
    add_subparsers(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    train_args = setup_run(args)

    # will download qm9 if necessary, calculate_triples is required for wACSF angular functions
    logging.info("QM9 will be loaded...")
    omdb = spk.datasets.OrganicMaterialsDatabase(
        args.datapath, args.cutoff, download=True, properties=[train_args.property]
    )
    atomref = omdb.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    if args.mode == "train":
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info("create splits...")
    data_train, data_val, data_test = omdb.create_splits(
        *train_args.split, split_file=split_path
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

    if args.mode == "train":
        logging.info("calculate statistics...")
        split_data = np.load(split_path)
        if "mean" in split_data.keys():
            mean = split_data["mean"].item()
            stddev = split_data["stddev"].item()
            calc_stats = False
            logging.info("cached statistics was loaded...")
        else:
            mean, stddev = train_loader.get_statistics(
                train_args.property, True, atomref
            )
            np.savez(
                split_path,
                train_idx=split_data["train_idx"],
                val_idx=split_data["val_idx"],
                test_idx=split_data["test_idx"],
                mean=mean,
                stddev=stddev,
            )

        # construct the model
        representation = get_representation(
            train_args,
            train_loader=train_loader,
            mode=args.mode
        )

        if args.model == "schnet":
            output_module = schnetpack.output_modules.Atomwise(
                args.features,
                aggregation_mode=args.aggregation_mode,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                train_embeddings=True,
                property=args.property,
            )
        else:
            raise NotImplementedError

        model = get_model(
            representation=representation,
            output_modules=output_module,
            parallelize=args.parallel,
        )

    metrics = [
        spk.metrics.MeanAbsoluteError(train_args.property, train_args.property),
        spk.metrics.RootMeanSquaredError(train_args.property, train_args.property),
    ]

    if args.mode == "train":
        logging.info("training...")
        train(args, model, train_loader, val_loader, device, metrics=metrics)
        logging.info("...training done!")
    elif args.mode == "eval":
        model = torch.load(os.path.join(args.modelpath, "best_model"))
        logging.info("evaluating...")
        test_loader = spk.data.AtomsLoader(
            data_test, batch_size=args.batch_size, num_workers=2, pin_memory=True
        )
        with torch.no_grad():
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
            )
        logging.info("... done!")
    else:
        print("Unknown mode:", args.mode)
