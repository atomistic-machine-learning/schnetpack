import os
import logging
from shutil import rmtree
import schnetpack as spk
from schnetpack.utils import to_json, read_from_json


def setup_run(args, from_file="False"):
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, "args.json")
    if args.mode == "train":
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info("existing model will be overwritten...")
            rmtree(args.modelpath)

        if not args.overwrite and from_file:
            if os.path.exists(jsonpath):
                logging.info("args.json file will be read to set training.")
                train_args = read_from_json(jsonpath)
                spk.utils.set_random_seed(args.seed)
                return train_args
            else:
                logging.info("args.json file was not found. Reset to default behavior.")
                os.makedirs(args.modelpath)
                to_json(jsonpath, argparse_dict)

        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        to_json(jsonpath, argparse_dict)

        spk.utils.set_random_seed(args.seed)
        train_args = args
    else:
        train_args = read_from_json(jsonpath)
    return train_args
