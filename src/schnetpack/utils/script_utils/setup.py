import os
import logging
from shutil import rmtree
import schnetpack as spk


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
        spk.utils.spk_utils.to_json(jsonpath, argparse_dict)

        spk.utils.spk_utils.set_random_seed(args.seed)
        train_args = args
    else:
        # check if modelpath is valid
        if not os.path.exists(args.modelpath):
            raise spk.utils.ScriptError(
                "The selected modeldir does not exist " "at {}!".format(args.modelpath)
            )

        # load training arguments
        train_args = spk.utils.spk_utils.read_from_json(jsonpath)

    # apply alias definitions
    train_args = apply_aliases(train_args)
    return train_args


def apply_aliases(args):
    # force alias for custom dataset
    if args.dataset == "custom":
        if args.force is not None:
            if args.derivative is not None:
                raise spk.utils.ScriptError(
                    "Force and derivative define the same property. Please don`t use "
                    "both."
                )
            args.derivative = args.force
            args.negative_dr = True

            # add rho value if selected
            if "force" in args.rho.keys():
                args.rho["derivative"] = args.rho.pop("force")

    return args
