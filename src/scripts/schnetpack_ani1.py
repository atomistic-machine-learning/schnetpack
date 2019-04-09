#!/usr/bin/env python
import logging
import os
from shutil import copyfile
import argparse

import numpy as np
import schnetpack.output_modules
import torch
from scripts.script_utils.evaluation import evaluate
from scripts.script_utils.training import train
from torch.utils.data.sampler import RandomSampler
from ase.data import atomic_numbers

import schnetpack as spk
from scripts.script_utils.model import get_representation, get_model
from scripts.script_utils.setup import setup_run
from scripts.script_utils.script_parsing import get_parser
from schnetpack.datasets import ANI1
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def ani1_parser(main_parser):
    ani1_parser = argparse.ArgumentParser(add_help=False, parents=[main_parser])
    ani1_parser.add_argument("datapath", help="Path / destination of dataset")
    ani1_parser.add_argument("modelpath", help="Destination for models and logs")
    ani1_parser.add_argument(
        "--property",
        type=str,
        help="Property to be predicted (default: %(default)s)",
        default="energy",
        choices=ANI1.available_properties,
    )
    return main_parser


if __name__ == "__main__":
    # get parser
    parser = get_parser()
    parser = ani1_parser(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # get train args from file or dump them
    train_args = setup_run(args)

    # will download ANI1 if necessary, calculate_triples is required for
    # wACSF angular functions
    logging.info("ANI1 will be loaded...")
    ani1 = spk.datasets.ANI1(
        args.datapath,
        download=True,
        properties=[train_args.property],
        collect_triples=args.model == "wacsf",
        num_heavy_atoms=2
    )
    atomref = ani1.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    if args.mode == "train":
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info("create splits...")
    data_train, data_val, data_test = ani1.create_splits(
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

    metrics = [
        spk.metrics.MeanAbsoluteError(train_args.property, train_args.property),
        spk.metrics.RootMeanSquaredError(train_args.property, train_args.property),
    ]

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
            output_modules = schnetpack.output_modules.Atomwise(
                args.features,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                aggregation_mode=args.aggregation_mode,
                property='energy'
            )
        elif args.model == "wacsf":
            # Build HDNN model
            elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
            output_modules = spk.output_modules.ElementalAtomwise(
                n_in=representation.n_symfuncs,
                n_hidden=args.n_nodes,
                n_layers=args.n_layers,
                mean=mean[args.property],
                stddev=stddev[args.property],
                aggregation_mode=args.aggregation_mode,
                atomref=atomref[args.property],
                elements=elements,
                property='energy'
            )
        model = get_model(
            representation=representation,
            output_modules=output_modules,
            parallelize=args.parallel,
        )
        logging.info("training...")
        train(args, model, train_loader, val_loader, device, metrics)
        logging.info("...training done!")

    elif args.mode == "eval":
        model = torch.load(os.path.join(args.modelpath, "best_model"))
        logging.info("evaluating...")
        test_loader = spk.data.AtomsLoader(
            data_test, batch_size=args.batch_size, num_workers=4, pin_memory=True
        )
        with torch.no_grad():
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics,
            )
        logging.info("... done!")
    else:
        print("Unknown mode:", args.mode)
