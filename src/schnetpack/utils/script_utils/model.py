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


def get_output_module_by_str(module_str):
    if module_str == "atomwise":
        return spk.atomistic.Atomwise
    elif module_str == "elemental_atomwise":
        return spk.atomistic.ElementalAtomwise
    elif module_str == "dipole_moment":
        return spk.atomistic.DipoleMoment
    elif module_str == "elemental_dipole_moment":
        return spk.atomistic.ElementalDipoleMoment
    elif module_str == "polarizability":
        return spk.atomistic.Polarizability
    elif module_str == "electronic_spatial_sxtent":
        return spk.atomistic.ElectronicSpatialExtent
    else:
        raise spk.utils.ScriptError(
            "{} is not a valid output " "module!".format(module_str)
        )


def get_output_module(args, representation, mean, stddev, atomref):
    derivative = spk.utils.get_derivative(args)
    negative_dr = spk.utils.get_negative_dr(args)
    contributions = spk.utils.get_contributions(args)
    stress = spk.utils.get_stress(args)
    if args.dataset == "md17" and not args.ignore_forces:
        derivative = spk.datasets.MD17.forces
    output_module_str = spk.utils.get_module_str(args)
    if output_module_str == "dipole_moment":
        return spk.atomistic.output_modules.DipoleMoment(
            args.features,
            predict_magnitude=True,
            mean=mean[args.property],
            stddev=stddev[args.property],
            property=args.property,
            contributions=contributions,
        )
    elif output_module_str == "electronic_spatial_extent":
        return spk.atomistic.output_modules.ElectronicSpatialExtent(
            args.features,
            mean=mean[args.property],
            stddev=stddev[args.property],
            property=args.property,
            contributions=contributions,
        )
    elif output_module_str == "atomwise":
        return spk.atomistic.output_modules.Atomwise(
            args.features,
            aggregation_mode=spk.utils.get_pooling_mode(args),
            mean=mean[args.property],
            stddev=stddev[args.property],
            atomref=atomref[args.property],
            property=args.property,
            derivative=derivative,
            negative_dr=negative_dr,
            contributions=contributions,
            stress=stress,
        )
    elif output_module_str == "polarizability":
        return spk.atomistic.output_modules.Polarizability(
            args.features,
            aggregation_mode=spk.utils.get_pooling_mode(args),
            property=args.property,
        )
    elif output_module_str == "isotropic_polarizability":
        return spk.atomistic.output_modules.Polarizability(
            args.features,
            aggregation_mode=spk.utils.get_pooling_mode(args),
            property=args.property,
            isotropic=True,
        )
    # wacsf modules
    elif output_module_str == "elemental_dipole_moment":
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        return spk.atomistic.output_modules.ElementalDipoleMoment(
            representation.n_symfuncs,
            n_hidden=args.n_nodes,
            n_layers=args.n_layers,
            predict_magnitude=True,
            elements=elements,
            property=args.property,
        )
    elif output_module_str == "elemental_atomwise":
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
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
