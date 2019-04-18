#!/usr/bin/env python
import logging
import os
import torch
from ase.data import atomic_numbers

import schnetpack as spk
from schnetpack.datasets import MD17
from schnetpack.output_modules import Atomwise, ElementalAtomwise
from scripts.script_utils import (
    get_main_parser,
    add_subparsers,
    train,
    get_representation,
    get_model,
    evaluate,
    setup_run,
    get_statistics,
    get_loaders,
    tradeoff_loff_fn,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def add_md17_arguments(parser):
    parser.add_argument(
        "--molecule",
        type=str,
        help="Choose molecule inside the MD17 dataset",
        default="ethanol",
        choices=MD17.datasets_dict.keys(),
    )
    parser.add_argument(
        "--rho",
        type=float,
        help="Energy-force trade-off. For rho=0, use forces only. (default: %(default)s)",
        default=0.1,
    )


if __name__ == "__main__":
    # parse arguments
    parser = get_main_parser()
    add_md17_arguments(parser)
    add_subparsers(
        parser,
        defaults=dict(property=MD17.energy, elements=["H", "C", "O"]),
        choices=dict(property=MD17.available_properties),
    )
    args = parser.parse_args()
    train_args = setup_run(args)

    # set device
    device = torch.device("cuda" if args.cuda else "cpu")

    # define metrics
    metrics = [
        spk.metrics.MeanAbsoluteError(MD17.energy, MD17.energy),
        spk.metrics.RootMeanSquaredError(MD17.energy, MD17.energy),
        spk.metrics.MeanAbsoluteError(MD17.forces, MD17.forces, element_wise=True),
        spk.metrics.RootMeanSquaredError(MD17.forces, MD17.forces, element_wise=True),
    ]

    # build dataset
    logging.info("MD17 will be loaded...")
    md17 = MD17(
        args.datapath,
        args.molecule,
        download=True,
        collect_triples=args.model == "wacsf",
    )

    # get atomrefs
    atomref = md17.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        logging, args, dataset=md17, split_path=split_path
    )

    if args.mode == "train":
        # get statistics
        logging.info("calculate statistics...")
        mean, stddev = get_statistics(
            split_path, logging, train_loader, train_args, atomref
        )

        # build representation
        representation = get_representation(args, train_loader)

        # build output module
        if args.model == "schnet":
            output_module = spk.output_modules.Atomwise(
                args.features,
                aggregation_mode=args.aggregation_mode,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                property=args.property,
                derivative="forces",
                negative_dr=True,
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

        # build AtomisticModel
        model = get_model(
            representation, output_modules=output_module, parallelize=args.parallel
        )

        # run training
        logging.info("training...")
        loss_fn = tradeoff_loff_fn(args, "forces")
        train(
            args,
            model,
            train_loader,
            val_loader,
            device,
            metrics=metrics,
            loss_fn=loss_fn,
        )
        logging.info("...training done!")

    elif args.mode == "eval":

        # header for output file
        header = ["Energy MAE", "Energy RMSE", "Force MAE", "Force RMSE"]

        # load model
        model = torch.load(os.path.join(args.modelpath, "best_model"))

        # run evaluation
        logging.info("evaluating...")
        evaluate(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            metrics,
            custom_header=header,
        )
        logging.info("... done!")
    else:
        raise NotImplementedError("Unknown mode:", args.mode)
