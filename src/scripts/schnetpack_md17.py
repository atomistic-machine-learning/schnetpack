#!/usr/bin/env python
import logging
import os
import sys
from shutil import copyfile

import numpy as np
import torch
from ase.data import atomic_numbers
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
from schnetpack.datasets import MD17
from schnetpack.output_modules import Atomwise, ElementalAtomwise
from scripts.script_utils import get_main_parser, add_subparsers, train, \
    get_representation, get_model, evaluate, setup_run

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def add_md17_arguments(parser):
    parser.add_argument(
        "--property",
        type=str,
        help="Property to be predicted (default: %(default)s)",
        default=MD17.energy,
        choices=MD17.available_properties,
    )
    parser.add_argument(
        "--molecule",
        type=str,
        help="Choose molecule inside the MD17 dataset",
        default="ethanol",
        choices=MD17.datasets_dict.keys(),
    )



if __name__ == "__main__":
    parser = get_main_parser()
    add_md17_arguments(parser)
    add_subparsers(parser)
    args = parser.parse_args()
    train_args = setup_run(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    # will download md17 if necessary, calculate_triples is required for wACSF angular functions
    logging.info("MD17 will be loaded...")
    md17 = MD17(
        args.datapath,
        args.molecule,
        download=True,
        collect_triples=args.model == "wacsf",
    )

    atomref = md17.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    if args.mode == "train":
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info("create splits...")
    data_train, data_val, data_test = md17.create_splits(
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

    metrics = [
        spk.metrics.MeanAbsoluteError(MD17.energy, MD17.energy),
        spk.metrics.RootMeanSquaredError(MD17.energy, MD17.energy),
        spk.metrics.MeanAbsoluteError(MD17.forces, MD17.forces),
        spk.metrics.RootMeanSquaredError(MD17.forces, MD17.forces),
    ]

    header = [
        "Subset",
        "Energy MAE",
        "Energy RMSE",
        "Force MAE",
        "Force RMSE",
        "Force Length MAE",
        "Force Length RMSE",
        "Force Angle MAE",
        "Angle RMSE",
    ]

    if args.mode == "train":
        logging.info("calculate statistics...")
        split_data = np.load(split_path)
        if "mean" in split_data.keys():
            mean = torch.from_numpy(split_data["mean"])
            stddev = torch.from_numpy(split_data["stddev"])
            calc_stats = False
            logging.info("cached statistics was loaded...")
        else:
            mean, stddev = train_loader.get_statistics(MD17.energy, True)
            np.savez(
                split_path,
                train_idx=split_data["train_idx"],
                val_idx=split_data["val_idx"],
                test_idx=split_data["test_idx"],
                mean=mean,
                stddev=stddev,
            )
        representation = get_representation(args, train_loader, mode="train")
        if args.model == "schnet":
            output_module = spk.output_modules.Atomwise(
                args.features,
                aggregation_mode=args.aggregation_mode,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                property=args.property,
                derivative="forces"
            )
        elif args.model == "wascf":
            elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
            output_module = ElementalAtomwise(
                representation.n_symfuncs,
                n_hidden=args.n_nodes,
                n_layers=args.n_layers,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.porperty],
                derivative="forces",
                create_graph=True,
                elements=elements,
                property=args.property,
            )

        else:
            raise NotImplementedError
        model = get_model(representation, output_modules=output_module,
                          parallelize=args.parallel)


        logging.info("training...")

        train(args, model, train_loader, val_loader, device, metrics=metrics)
        logging.info("...training done!")
    elif args.mode == "eval":
        model = torch.load(os.path.join(args.modelpath, "best_model"))
        logging.info("evaluating...")
        test_loader = spk.data.AtomsLoader(
            data_test, batch_size=args.batch_size, num_workers=2, pin_memory=True
        )
        evaluate(args, model, train_loader, val_loader, test_loader, device,
                 metrics)
        logging.info("... done!")
    else:
        print("Unknown mode:", args.mode)
