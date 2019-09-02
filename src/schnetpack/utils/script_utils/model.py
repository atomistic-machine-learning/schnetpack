import torch
import logging
import schnetpack as spk
from ase.data import atomic_numbers
import torch.nn as nn

__all__ = ["get_representation", "get_output_module", "get_model"]


def get_representation(args, train_loader=None):
    # build representation
    if args.model == "schnet":

        cutoff_network = spk.nn.cutoff.get_cutoff_by_string(args.cutoff_function)

        return spk.representation.SchNet(
            n_atom_basis=args.features,
            n_filters=args.features,
            n_interactions=args.interactions,
            cutoff=args.cutoff,
            n_gaussians=args.num_gaussians,
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


def get_output_module(args, representation, mean, stddev, atomref):
    derivative = spk.utils.get_derivative(args)
    negative_dr = spk.utils.get_negative_dr(args)
    if args.dataset == "md17" and not args.ignore_forces:
        derivative = spk.datasets.MD17.forces
    if args.model == "schnet":
        if args.property == spk.datasets.QM9.mu:
            return spk.atomistic.output_modules.DipoleMoment(
                args.features,
                predict_magnitude=True,
                mean=mean[args.property],
                stddev=stddev[args.property],
                property=args.property,
            )
        elif args.property == spk.datasets.QM9.r2:
            return spk.atomistic.output_modules.ElectronicSpatialExtent(
                args.features,
                mean=mean[args.property],
                stddev=stddev[args.property],
                property=args.property,
            )
        else:
            return spk.atomistic.output_modules.Atomwise(
                args.features,
                aggregation_mode=spk.utils.get_pooling_mode(args),
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                property=args.property,
                derivative=derivative,
                negative_dr=negative_dr,
            )
    elif args.model == "wacsf":
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        if args.property == spk.datasets.QM9.mu:
            return spk.atomistic.output_modules.ElementalDipoleMoment(
                representation.n_symfuncs,
                n_hidden=args.n_nodes,
                n_layers=args.n_layers,
                predict_magnitude=True,
                elements=elements,
                property=args.property,
            )
        else:
            return spk.atomistic.output_modules.ElementalAtomwise(
                representation.n_symfuncs,
                n_hidden=args.n_nodes,
                n_layers=args.n_layers,
                aggregation_mode=spk.utils.get_pooling_mode(args),
                mean=mean[args.property],
                stddev=stddev[args.property],
                atomref=atomref[args.property],
                elements=elements,
                property=args.property,
                derivative=derivative,
                negative_dr=negative_dr,
            )
    else:
        raise NotImplementedError


def get_model(args, train_loader, mean, stddev, atomref, logging=None):
    """
    Build a model from selected parameters or load trained model for evaluation.

    Args:
        args (argsparse.Namespace): Script arguments
        train_loader (spk.AtomsLoader): loader for training data
        mean (torch.Tensor): mean of training data
        stddev (torch.Tensor): stddev of training data
        atomref (dict): atomic references
        logging: logger

    Returns:
        spk.AtomisticModel: model for training or evaluation
    """
    if args.mode == "train":
        if logging:
            logging.info("building model...")
        representation = get_representation(args, train_loader)
        output_module = get_output_module(
            args,
            representation=representation,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
        )
        model = spk.AtomisticModel(representation, [output_module])

        if args.parallel:
            model = nn.DataParallel(model)
        if logging:
            logging.info(
                "The model you built has: %d parameters" % spk.utils.count_params(model)
            )
        return model
    else:
        raise spk.utils.ScriptError("Invalid mode selected: {}".format(args.mode))
