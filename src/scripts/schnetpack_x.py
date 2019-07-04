import os
import torch
import logging
from shutil import rmtree
import schnetpack as spk
from schnetpack.utils import (
    get_dataset,
    get_metrics,
    get_loaders,
    get_statistics,
    get_representation,
    get_output_module,
    get_model,
    get_trainer,
    ScriptError,
    evaluate,
)
from schnetpack.utils.script_utils.parser_stuff import build_parser


def main(args):
    """

    Args:
        args:

    Returns:

    """
    # setup
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, "args.json")
    if args.mode == "train":
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info("existing model will be overwritten...")
            rmtree(args.modelpath)

        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        spk.utils.spk_utils.to_json(jsonpath, argparse_dict)

        spk.utils.spk_utils.set_random_seed(args.seed)
        train_args = args
    else:
        train_args = spk.utils.spk_utils.read_from_json(jsonpath)

    # get dataset
    dataset = get_dataset(args)

    # get dataloaders
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=dataset, split_path=split_path, logging=logging
    )

    # define metrics
    metrics = get_metrics(args)

    # train or evaluate
    if args.mode == "train":
        train(
            args,
            metrics,
            split_path,
            train_loader,
            val_loader,
            dataset,
            logging=logging,
        )
    elif args.mode == "eval":
        eval(args, train_loader, val_loader, test_loader, metrics)
    else:
        raise ScriptError("Unknown mode: {}".format(args.mode))


def train(args, metrics, split_path, train_loader, val_loader, dataset, logging=None):
    """

    Args:
        args:
        metrics:

    Returns:

    """
    # get statistics
    if logging:
        logging.info("calculate statistics...")
    atomref = dataset.get_atomrefs(args.property)
    mean, stddev = get_statistics(
        split_path, train_loader, args, atomref, logging=logging
    )

    # build model
    representation = get_representation(args, train_loader)
    output_module = get_output_module(
        args, representation=representation, mean=mean, stddev=stddev, atomref=atomref
    )
    model = get_model(
        args, representation=representation, output_modules=[output_module]
    )

    # build trainer
    logging.info("training...")
    trainer = get_trainer(args, model, train_loader, val_loader, metrics)

    # run training
    device = torch.device("cuda" if args.cuda else "cpu")
    trainer.train(device, n_epochs=args.n_epochs)
    logging.info("...training done!")


def eval(args, train_loader, val_loader, test_loader, metrics, logging=None):
    """

    Args:
        args:
        metrics:

    Returns:

    """
    # load model
    if logging:
        logging.info("loading model...")
    model = torch.load(os.path.join(args.modelpath, "best_model"))

    # run evaluation
    if logging:
        logging.info("evaluating...")
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        evaluate(
            args, model, train_loader, val_loader, test_loader, device, metrics=metrics
        )
    logging.info("... evaluation done!")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print("*", args)
    main(args)
