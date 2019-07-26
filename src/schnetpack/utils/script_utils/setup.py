import os
import logging
from shutil import rmtree
import schnetpack.utils.spk_utils

__all__ = ["setup_run"]


def setup_run(args):
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, "args.json")
    if args.mode == "train":

        # build modeldir
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info("existing model will be overwritten...")
            rmtree(args.modelpath)
        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        # store training arguments
        schnetpack.utils.spk_utils.to_json(jsonpath, argparse_dict)

        schnetpack.utils.spk_utils.set_random_seed(args.seed)
        train_args = args
    else:
        # check if modelpath is valid
        if not os.path.exists(args.modelpath):
            raise schnetpack.utils.ScriptError(
                "The selected modeldir does not exist " "at {}!".format(args.modelpath)
            )

        # load training arguments
        train_args = schnetpack.utils.spk_utils.read_from_json(jsonpath)

    return train_args
