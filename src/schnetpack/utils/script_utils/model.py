import logging
import schnetpack as spk
from ase.data import atomic_numbers
import torch.nn as nn
import schnetpack.utils as utils
from schnetpack.atomistic import AtomisticModel
from schnetpack.nn.cutoff import get_cutoff_by_string


__all__ = ["get_representation", "get_model"]


def get_representation(args, train_loader=None):
    # build representation
    if args.model == "schnet":

        cutoff_network = get_cutoff_by_string(args.cutoff_function)

        return spk.representation.SchNet(
            args.features,
            args.n_filters,
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

    logging.info("The model you built has: %d parameters" % utils.compute_params(model))

    return model
