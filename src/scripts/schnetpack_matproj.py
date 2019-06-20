#!/usr/bin/env python
import logging
import os
import torch

import schnetpack as spk
from schnetpack.datasets import MaterialsProject
from scripts.script_utils import (
    get_representation,
    get_model,
    setup_run,
    get_trainer,
    evaluate,
    get_main_parser,
    add_subparsers,
    get_statistics,
    get_loaders,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def add_matproj_arguments(parser):
    parser.add_argument(
        "--apikey",
        help="API key for Materials Project (see https://materialsproject.org/open)",
        default=None,
    )


if __name__ == "__main__":
    # parse arguments
    parser = get_main_parser()
    add_matproj_arguments(parser)
    add_subparsers(
        parser,
        defaults=dict(
            property=MaterialsProject.EformationPerAtom,
            features=64,
            aggregation_mode="mean",
        ),
        choices=dict(
            property=[
                MaterialsProject.EformationPerAtom,
                MaterialsProject.EPerAtom,
                MaterialsProject.BandGap,
                MaterialsProject.TotalMagnetization,
            ]
        ),
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
    mp = spk.datasets.MaterialsProject(
        args.datapath,
        args.cutoff,
        apikey=args.apikey,
        download=True,
        properties=[train_args.property],
    )

    # get atomrefs
    atomref = mp.get_atomrefs(train_args.property)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=mp, split_path=split_path, logging=logging
    )

    if args.mode == "train":
        # get statistics
        logging.info("calculate statistics...")
        mean, stddev = get_statistics(
            split_path, train_loader, train_args, atomref, logging=logging
        )

        # build representation
        representation = get_representation(train_args, train_loader=train_loader)

        # build output module
        if args.model == "schnet":
            output_module = spk.output_modules.Atomwise(
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

        # build AtomisticModel
        model = get_model(
            representation=representation,
            output_modules=output_module,
            parallelize=args.parallel,
        )

        # run training
        logging.info("Training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        trainer.train(device, n_epochs=args.n_epochs)
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
                train_args.property,
                train_loader,
                val_loader,
                test_loader,
                device,
            )
    else:
        raise NotImplementedError("Unknown mode:", args.mode)
