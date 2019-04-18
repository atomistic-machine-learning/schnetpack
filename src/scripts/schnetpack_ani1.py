#!/usr/bin/env python
import logging
import os
import torch

import schnetpack.output_modules
from scripts.script_utils.evaluation import evaluate
from scripts.script_utils.training import train
from ase.data import atomic_numbers

import schnetpack as spk
from schnetpack.datasets import ANI1
from scripts.script_utils import (
    setup_run,
    get_representation,
    get_model,
    get_main_parser,
    add_subparsers,
    get_loaders,
    get_statistics,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # parse arguments
    parser = get_main_parser()
    add_subparsers(
        parser,
        defaults=dict(property=ANI1.energy),
        choices=dict(property=ANI1.available_properties),
    )
    args = parser.parse_args()
    train_args = setup_run(args)

    # set device
    device = torch.device("cuda" if args.cuda else "cpu")

    # define metrics
    metrics = [
        spk.metrics.MeanAbsoluteError(train_args.property, train_args.property),
        spk.metrics.RootMeanSquaredError(train_args.property, train_args.property),
    ]

    # build dataset
    logging.info("ANI1 will be loaded...")
    ani1 = spk.datasets.ANI1(
        args.datapath,
        download=True,
        properties=[train_args.property],
        collect_triples=args.model == "wacsf",
        # todo: remove
        num_heavy_atoms=2,
    )

    # get atomrefs
    atomref = ani1.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        logging, args, dataset=ani1, split_path=split_path
    )

    if args.mode == "train":
        # get statistics
        logging.info("calculate statistics...")
        mean, stddev = get_statistics(
            split_path, logging, train_loader, train_args, atomref
        )

        # build representation
        representation = get_representation(train_args, train_loader=train_loader)

        # build output module
        if args.model == "schnet":
            output_modules = schnetpack.output_modules.Atomwise(
                args.features,
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                aggregation_mode=args.aggregation_mode,
                property="energy",
            )
        elif args.model == "wacsf":
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
                property="energy",
            )
        else:
            raise NotImplementedError("Model {} is not known".format(args.model))

        # build AtomisticModel
        model = get_model(
            representation=representation,
            output_modules=output_modules,
            parallelize=args.parallel,
        )

        # run training
        logging.info("training...")
        train(args, model, train_loader, val_loader, device, metrics)
        logging.info("...training done!")

    elif args.mode == "eval":
        # load model
        model = torch.load(os.path.join(args.modelpath, "best_model"))

        # run evaluation
        logging.info("evaluating...")
        with torch.no_grad():
            evaluate(
                args, model, train_loader, val_loader, test_loader, device, metrics
            )
        logging.info("... done!")
    else:
        raise NotImplementedError("Unknown mode:", args.mode)
