#!/usr/bin/env python
import logging
import os
import sys
from shutil import copyfile, rmtree

import numpy as np
import schnetpack.output_modules
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
from schnetpack.datasets import MaterialsProject
from schnetpack.utils import to_json, read_from_json, compute_params
from scripts.script_utils.model import get_representation, get_model
from scripts.script_utils.setup import setup_run
from scripts.script_utils.training import train
from scripts.script_utils.evaluation import evaluate
from scripts.script_utils.script_parsing import get_main_parser, add_subparsers

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def add_matproj_arguments(parser):
    parser.add_argument(
        "--apikey",
        help="API key for Materials Project (see https://materialsproject.org/open)",
        default=None,
    )
    parser.add_argument(
        "--property",
        type=str,
        help="Materials Project property to be predicted (default: %(default)s)",
        default="formation_energy_per_atom",
        choices=MaterialsProject.available_properties,
    )


if __name__ == "__main__":
    parser = get_main_parser()
    add_matproj_arguments(parser)
    add_subparsers(parser)
    args = parser.parse_args()
    train_args = setup_run(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    # will download MaterialsProject if necessary
    mp = spk.datasets.MaterialsProject(
        args.datapath,
        args.cutoff,
        apikey=args.apikey,
        download=True,
        properties=[train_args.property],
    )

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    if args.mode == "train":
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    data_train, data_val, data_test = mp.create_splits(
        *train_args.split, split_file=split_path
    )

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
        spk.metrics.MeanAbsoluteError(args.property, args.property),
        spk.metrics.RootMeanSquaredError(args.property, args.property),
    ]

    if args.mode == "train":
        logging.info("calculate statistics...")
        split_data = np.load(split_path)
        #if "mean" in split_data.keys():
        #    mean = split_data["mean"].item()
        #    stddev = split_data["stddev"].item()
        #    calc_stats = False
        #    logging.info("cached statistics was loaded...")
        #else:
        #    mean, stddev = train_loader.get_statistics(
        #        train_args.property, True
        #    )  # , atomref)
        #    np.savez(
        #        split_path,
        #        train_idx=split_data["train_idx"],
        #        val_idx=split_data["val_idx"],
        #        test_idx=split_data["test_idx"],
        #        mean=mean,
        #        stddev=stddev,
        #    )
        mean = {args.property: None}
        stddev = {args.property: None}
        representation = get_representation(args, train_loader=train_loader)
        if args.model == "schnet":
            atomwise_output = schnetpack.output_modules.Atomwise(
                args.features,
                aggregation_mode=args.aggregation_mode,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=None,
                train_embeddings=True,
                property=args.property,
            )
        else:
            raise NotImplementedError

        model = get_model(representation=representation,
                          output_modules=atomwise_output,
                          parallelize=args.parallel)
        logging.info("Training...")
        train(args, model, train_loader, val_loader, device, metrics=metrics)

    elif args.mode == "eval":
        model = torch.load(os.path.join(args.modelpath, "best_model"))

        test_loader = spk.data.AtomsLoader(
            data_test, batch_size=args.batch_size, num_workers=2, pin_memory=True
        )
        evaluate(
            args,
            model,
            train_args.property,
            train_loader,
            val_loader,
            test_loader,
            device,
        )
    else:
        raise NotImplementedError
