import logging
import schnetpack as spk
from ase.data import atomic_numbers
import torch.nn as nn
from schnetpack.utils import compute_params
from schnetpack.atomistic import AtomisticModel
from schnetpack.nn.cutoff import HardCutoff, MollifierCutoff, CosineCutoff


def get_representation(args, train_loader=None):
    # build representation
    if args.model == "schnet":

        # build cutoff module
        if args.cutoff_function == "hard":
            cutoff_network = HardCutoff
        elif args.cutoff_function == "cosine":
            cutoff_network = CosineCutoff
        elif args.cutoff_function == "mollifier":
            cutoff_network = MollifierCutoff
        else:
            raise NotImplementedError(
                "cutoff_function {} is unknown".format(args.cutoff_function)
            )

        return spk.representation.SchNet(
            args.features,
            args.features,
            args.interactions,
            args.cutoff,
            args.num_gaussians,
            cutoff_network=cutoff_network,
        )

    elif args.model == "wacsf":
        sfmode = ("weighted", "Behler")[args.behler]
        # Convert element strings to atomic charges
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        representation = spk.representation.BehlerSFBlock(
            args.radial,
            args.angular,
            zetas=set(args.zetas),
            cutoff_radius=args.cutoff,
            centered=args.centered,
            crossterms=args.crossterms,
            elements=elements,
            mode=sfmode,
        )
        logging.info(
            "Using {:d} {:s}-type SF".format(representation.n_symfuncs, sfmode)
        )
        # Standardize representation if requested
        if args.standardize:
            if train_loader is None:
                raise ValueError(
                    "Specification of a training_loader is required to standardize "
                    "wACSF"
                )
            else:
                logging.info("Computing and standardizing symmetry function statistics")
                return spk.representation.StandardizeSF(
                    representation, train_loader, cuda=args.cuda
                )

        else:
            return representation

    else:
        raise NotImplementedError("Unknown model class:", args.model)


def get_model(representation, output_modules, parallelize=False):

    model = AtomisticModel(representation, output_modules)

    if parallelize:
        model = nn.DataParallel(model)

    logging.info("The model you built has: %d parameters" % compute_params(model))

    return model
