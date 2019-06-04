#!/usr/bin/env python
import logging
import os

import schnetpack.output_modules
import torch
from ase.data import atomic_numbers

import schnetpack as spk
from schnetpack.datasets import QM9
from scripts.script_utils.script_parsing import get_main_parser, add_subparsers
from scripts.script_utils.setup import setup_run
from scripts.script_utils.model import get_representation, get_model
from scripts.script_utils.training import train
from scripts.script_utils.evaluation import evaluate
from scripts.script_utils import get_statistics, get_loaders

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def add_qm9_arguments(base_parser):
    base_parser.add_argument(
        "--remove_uncharacterized",
        type=bool,
        help="Remove uncharacterized molecules from QM9",
        default=False,
    )


if __name__ == "__main__":
    # parse arguments
    parser = get_main_parser()
    add_qm9_arguments(parser)
    add_subparsers(
        parser,
        defaults=dict(property=QM9.U0),
        choices=dict(property=QM9.available_properties),
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
    logging.info("QM9 will be loaded...")
    qm9 = QM9(
        args.datapath,
        download=True,
        properties=[train_args.property],
        collect_triples=args.model == "wacsf",
        remove_uncharacterized=train_args.remove_uncharacterized,
    )

    # get atomrefs
    atomref = qm9.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        logging, args, dataset=qm9, split_path=split_path
    )

    if args.mode == "train":
        # get statistics
        logging.info("calculate statistics...")
        mean, stddev = get_statistics(
            split_path, logging, train_loader, train_args, atomref
        )

        # build representation
        representation = get_representation(args, train_loader=train_loader)

        # build output module
        if args.model == "schnet":
            if args.property == QM9.mu:
                output_module = spk.output_modules.DipoleMoment(
                    args.features,
                    predict_magnitude=True,
                    mean=mean[args.property],
                    stddev=stddev[args.property],
                    property=args.property,
                )
            else:
                output_module = spk.output_modules.Atomwise(
                    args.features,
                    aggregation_mode=args.aggregation_mode,
                    mean=mean[args.property],
                    stddev=stddev[args.property],
                    atomref=atomref[args.property],
                    property=args.property,
                )
        elif args.model == "wacsf":
            elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
            if args.property == QM9.mu:
                output_module = spk.output_modules.ElementalDipoleMoment(
                    representation.n_symfuncs,
                    n_hidden=args.n_nodes,
                    n_layers=args.n_layers,
                    predict_magnitude=True,
                    elements=elements,
                    property=args.property,
                )
            else:
                output_module = spk.output_modules.ElementalAtomwise(
                    representation.n_symfuncs,
                    n_hidden=args.n_nodes,
                    n_layers=args.n_layers,
                    aggregation_mode=args.aggregation_mode,
                    mean=mean[args.property],
                    stddev=stddev[args.property],
                    atomref=atomref[args.property],
                    elements=elements,
                    property=args.property,
                )
        else:
            raise NotImplementedError

        # build AtomisticModel
        model = get_model(
            representation=representation,
            output_modules=output_module,
            parallelize=args.parallel,
        )

        # run training
        logging.info("training...")
        train(args, model, train_loader, val_loader, device, metrics=metrics)
        logging.info("...training done!")

    elif args.mode == "eval":
        # load model
        model = torch.load(os.path.join(args.modelpath, "best_model"))

        # run evaluation
        logging.info("evaluating...")
        with torch.no_grad():
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
                to_kcal=True,
            )
        logging.info("... done!")
    else:
        raise NotImplementedError("Unknown mode:", args.mode)
