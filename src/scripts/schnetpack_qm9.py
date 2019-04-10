#!/usr/bin/env python
import logging
import os
from shutil import copyfile

import numpy as np
import schnetpack.output_modules
import torch
from ase.data import atomic_numbers
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
from schnetpack.datasets import QM9
from scripts.script_utils.script_parsing import main_parser
from scripts.script_utils.setup import setup_run
from scripts.script_utils.model import get_representation, get_model
from scripts.script_utils.training import train
from scripts.script_utils.evaluation import evaluate
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def qm9_parser(main_parser):
    main_parser.add_argument(
        "--property",
        type=str,
        help="Property to be predicted (default: %(default)s)",
        default=QM9.U0,
        choices=QM9.available_properties,
    )
    main_parser.add_argument(
        "--remove_uncharacterized",
        type=bool,
        help="Remove uncharacterized molecules from QM9",
        default=False,
    )
    return main_parser


if __name__ == "__main__":
    parser = main_parser
    parser = qm9_parser(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    train_args = setup_run(args)

    # will download qm9 if necessary, calculate_triples is required for wACSF angular functions
    logging.info("QM9 will be loaded...")
    qm9 = QM9(
        args.datapath,
        download=True,
        properties=[train_args.property],
        collect_triples=args.model == "wacsf",
        remove_uncharacterized=train_args.remove_uncharacterized,
    )
    atomref = qm9.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    if args.mode == "train":
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info("create splits...")
    data_train, data_val, data_test = qm9.create_splits(
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
            if args.property == QM9.mu:
                output_module = spk.output_modules.DipoleMoment(
                    args.features, predict_magnitude=True, mean=mean[args.property],
                    stddev=stddev[args.property]
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
        elif args.model == "wascf":
            elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
            if args.property == QM9.mu:
                output_module = spk.output_modules.ElementalDipoleMoment(
                    representation.n_symfuncs,
                    n_hidden=args.n_nodes,
                    n_layers=args.n_layers,
                    predict_magnitude=True,
                    elements=elements,
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
