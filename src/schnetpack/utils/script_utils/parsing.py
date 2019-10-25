import argparse
from schnetpack.datasets import (
    QM9,
    ANI1,
    MD17,
    OrganicMaterialsDatabase,
    MaterialsProject,
)


class StoreDictKeyPair(argparse.Action):
    """
    From https://stackoverflow.com/a/42355279
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def get_mode_parsers():
    mode_parser = argparse.ArgumentParser(add_help=False)

    # mode parsers

    # json parser
    json_parser = argparse.ArgumentParser(add_help=False, parents=[mode_parser])
    json_parser.add_argument(
        "json_path",
        type=str,
        help="Path to argument file. (default: %(default)s)",
        default=None,
    )

    # train parser
    train_parser = argparse.ArgumentParser(add_help=False, parents=[mode_parser])
    train_parser.add_argument("datapath", help="Path to dataset")
    train_parser.add_argument("modelpath", help="Path of stored model")
    train_parser.add_argument(
        "--cuda", help="Set flag to use GPU(s) for training", action="store_true"
    )
    train_parser.add_argument(
        "--parallel",
        help="Run data-parallel on all available GPUs (specify with environment"
        " variable CUDA_VISIBLE_DEVICES)",
        action="store_true",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for training (default: %(default)s)",
        default=100,
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="Set random seed for torch and numpy."
    )
    train_parser.add_argument(
        "--overwrite", help="Remove previous model directory.", action="store_true"
    )

    # data split
    train_parser.add_argument(
        "--split_path", help="Path / destination of npz with data splits", default=None
    )
    train_parser.add_argument(
        "--split",
        help="Split into [train] [validation] and use remaining for testing",
        type=int,
        nargs=2,
        default=[None, None],
    )
    train_parser.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of training epochs (default: %(default)s)",
        default=5000,
    )
    train_parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of training steps (default: %(default)s)",
        default=None,
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        help="Initial learning rate (default: %(default)s)",
        default=1e-4,
    )
    train_parser.add_argument(
        "--lr_patience",
        type=int,
        help="Epochs without improvement before reducing the learning rate "
        "(default: %(default)s)",
        default=25,
    )
    train_parser.add_argument(
        "--lr_decay",
        type=float,
        help="Learning rate decay (default: %(default)s)",
        default=0.8,
    )
    train_parser.add_argument(
        "--lr_min",
        type=float,
        help="Minimal learning rate (default: %(default)s)",
        default=1e-6,
    )

    train_parser.add_argument(
        "--logger",
        help="Choose logger for training process (default: %(default)s)",
        choices=["csv", "tensorboard"],
        default="csv",
    )
    train_parser.add_argument(
        "--log_every_n_epochs",
        type=int,
        help="Log metrics every given number of epochs (default: %(default)s)",
        default=1,
    )
    train_parser.add_argument(
        "--n_epochs",
        type=int,
        help="Maximum number of training epochs (default: %(default)s)",
        default=1000,
    )
    train_parser.add_argument(
        "--checkpoint_interval",
        type=int,
        help="Store checkpoint every n epochs (default: %(default)s)",
        default=1,
    )
    train_parser.add_argument(
        "--keep_n_checkpoints",
        type=int,
        help="Number of checkpoints that will be stored (default: %(default)s)",
        default=3,
    )

    # evaluation parser
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[mode_parser])
    eval_parser.add_argument("datapath", help="Path to dataset")
    eval_parser.add_argument("modelpath", help="Path of stored model")
    eval_parser.add_argument(
        "--cuda", help="Set flag to use GPU(s) for evaluation", action="store_true"
    )
    eval_parser.add_argument(
        "--parallel",
        help="Run data-parallel on all available GPUs (specify with environment"
        " variable CUDA_VISIBLE_DEVICES)",
        action="store_true",
    )
    eval_parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for evaluation (default: %(default)s)",
        default=100,
    )
    eval_parser.add_argument(
        "--split",
        help="Evaluate trained model on given split",
        choices=["train", "validation", "test"],
        default=["test"],
        nargs="+",
    )
    eval_parser.add_argument(
        "--overwrite", help="Remove previous evaluation files", action="store_true"
    )

    return mode_parser, json_parser, train_parser, eval_parser


def get_model_parsers():
    # model parsers
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Cutoff radius of local environment (default: %(default)s)",
    )
    schnet_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    schnet_parser.add_argument(
        "--features", type=int, help="Size of atom-wise representation", default=128
    )
    schnet_parser.add_argument(
        "--interactions", type=int, help="Number of interaction blocks", default=6
    )
    schnet_parser.add_argument(
        "--cutoff_function",
        help="Functional form of the cutoff",
        choices=["hard", "cosine", "mollifier"],
        default="cosine",
    )
    schnet_parser.add_argument(
        "--num_gaussians",
        type=int,
        default=50,
        help="Number of Gaussians to expand distances (default: %(default)s)",
    )

    wacsf_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    wacsf_parser.add_argument(
        "--radial",
        type=int,
        default=22,
        help="Number of radial symmetry functions (default: %(default)s)",
    )
    wacsf_parser.add_argument(
        "--angular",
        type=int,
        default=5,
        help="Number of angular symmetry functions (default: %(default)s)",
    )
    wacsf_parser.add_argument(
        "--zetas",
        type=int,
        nargs="+",
        default=[1],
        help="List of zeta exponents used for angle resolution (default: %(default)s)",
    )
    wacsf_parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize wACSF before atomistic network.",
    )
    # Atomistic network parameters
    wacsf_parser.add_argument(
        "--n_nodes",
        type=int,
        default=100,
        help="Number of nodes in atomic networks (default: %(default)s)",
    )
    wacsf_parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in atomic networks (default: %(default)s)",
    )
    # Advances wACSF settings
    wacsf_parser.add_argument(
        "--centered",
        action="store_true",
        help="Use centered Gaussians for radial functions",
    )
    wacsf_parser.add_argument(
        "--crossterms", action="store_true", help="Use crossterms in angular functions"
    )
    wacsf_parser.add_argument(
        "--behler", action="store_true", help="Switch to conventional ACSF"
    )
    wacsf_parser.add_argument(
        "--elements",
        default=["H", "C", "N", "O", "F"],
        nargs="+",
        help="List of elements to be used for symmetry functions "
        "(default: %(default)s).",
    )

    return model_parser, schnet_parser, wacsf_parser


def get_data_parsers():
    # data parsers
    data_parser = argparse.ArgumentParser(add_help=False)
    data_parser.add_argument(
        "--environment_provider",
        type=str,
        default="simple",
        choices=["simple", "ase", "torch"],
        help="Environment provider for dataset. (default: %(default)s)",
    )

    # qm9
    qm9_parser = argparse.ArgumentParser(add_help=False, parents=[data_parser])
    qm9_parser.add_argument(
        "--property",
        type=str,
        help="Database property to be predicted (default: %(default)s)",
        default=QM9.U0,
        choices=[
            QM9.A,
            QM9.B,
            QM9.C,
            QM9.mu,
            QM9.alpha,
            QM9.homo,
            QM9.lumo,
            QM9.gap,
            QM9.r2,
            QM9.zpve,
            QM9.U0,
            QM9.U,
            QM9.H,
            QM9.G,
            QM9.Cv,
        ],
    )
    qm9_parser.add_argument(
        "--remove_uncharacterized",
        help="Remove uncharacterized molecules from QM9 (default: %(default)s)",
        action="store_true",
    )

    ani1_parser = argparse.ArgumentParser(add_help=False, parents=[data_parser])
    ani1_parser.add_argument(
        "--property",
        type=str,
        help="Database property to be predicted (default: %(default)s)",
        default=ANI1.energy,
        choices=[ANI1.energy],
    )
    ani1_parser.add_argument(
        "--num_heavy_atoms",
        type=int,
        help="Number of heavy atoms that will be loaded into the database."
        " (default: %(default)s)",
        default=8,
    )
    matproj_parser = argparse.ArgumentParser(add_help=False, parents=[data_parser])
    matproj_parser.add_argument(
        "--property",
        type=str,
        help="Database property to be predicted" " (default: %(default)s)",
        default=MaterialsProject.EformationPerAtom,
        choices=[
            MaterialsProject.EformationPerAtom,
            MaterialsProject.EPerAtom,
            MaterialsProject.BandGap,
            MaterialsProject.TotalMagnetization,
        ],
    )
    matproj_parser.add_argument(
        "--apikey",
        help="API key for Materials Project (see https://materialsproject.org/open)",
        default=None,
    )
    md17_parser = argparse.ArgumentParser(add_help=False, parents=[data_parser])
    md17_parser.add_argument(
        "--property",
        type=str,
        help="Database property to be predicted" " (default: %(default)s)",
        default=MD17.energy,
        choices=[MD17.energy],
    )
    md17_parser.add_argument(
        "--ignore_forces", action="store_true", help="Ignore forces during training."
    )
    md17_parser.add_argument(
        "--molecule",
        type=str,
        help="Choose molecule inside the MD17 dataset. (default: %(default)s)",
        default="ethanol",
        choices=MD17.datasets_dict.keys(),
    )
    md17_parser.add_argument(
        "--rho",
        type=float,
        help="Energy-force trade-off. For rho=0, use forces only. "
        "(default: %(default)s)",
        default=0.1,
    )
    omdb_parser = argparse.ArgumentParser(add_help=False, parents=[data_parser])
    omdb_parser.add_argument(
        "--property",
        type=str,
        help="Database property to be predicted (default: %(default)s)",
        default=OrganicMaterialsDatabase.BandGap,
        choices=[OrganicMaterialsDatabase.BandGap],
    )
    custom_data_parser = argparse.ArgumentParser(add_help=False, parents=[data_parser])
    custom_data_parser.add_argument(
        "--property",
        type=str,
        help="Database property to be predicted (default: %(default)s)",
        default="energy",
    )
    custom_data_parser.add_argument(
        "--derivative",
        type=str,
        help="Derivative of dataset property to be predicted (default: %(default)s)",
        default=None,
    )
    custom_data_parser.add_argument(
        "--negative_dr",
        action="store_true",
        help="Multiply derivatives with -1 for training. (default: %(default)s)",
    )
    custom_data_parser.add_argument(
        "--force",
        type=str,
        help="Name of force property in database. Alias for‚ derivative + setting "
        "negative_dr. (default: %(default)s)",
        default=None,
    )
    custom_data_parser.add_argument(
        "--contributions",
        type=str,
        help="Contributions of dataset property to be predicted (default: %(default)s)",
        default=None,
    )
    custom_data_parser.add_argument(
        "--stress",
        type=str,
        help="Train on stress tensor if not None (default: %(default)s)",
        default=None,
    )
    custom_data_parser.add_argument(
        "--aggregation_mode",
        type=str,
        help="Select mode for aggregating atomic properties. (default: %(default)s)",
        default="sum",
    )
    custom_data_parser.add_argument(
        "--output_module",
        type=str,
        help="Select matching output module for selected property. (default: %("
        "default)s)",
        default="atomwise",
        choices=[
            "atomwise",
            "elemental_atomwise",
            "dipole_moment",
            "elemental_dipole_moment",
            "polarizability",
            "isotropic_polarizability",
            "electronic_spatial_extent",
        ],
    )
    custom_data_parser.add_argument(
        "--rho",
        action=StoreDictKeyPair,
        nargs="+",
        metavar="KEY=VAL",
        help="Define loss tradeoff weights with prop=weight. (default: %(default)s)",
        default=dict(),
    )

    return (
        data_parser,
        qm9_parser,
        ani1_parser,
        matproj_parser,
        md17_parser,
        omdb_parser,
        custom_data_parser,
    )


def build_parser():
    # get parsers
    mode_parser, json_parser, train_parser, eval_parser = get_mode_parsers()
    model_parser, schnet_parser, wacsf_parser = get_model_parsers()
    data_parser, qm9_parser, ani1_parser, matproj_parser, md17_parser, omdb_parser, custom_data_parser = (
        get_data_parsers()
    )

    # subparser structure
    # mode
    mode_subparsers = mode_parser.add_subparsers(dest="mode", help="main arguments")
    train_subparser = mode_subparsers.add_parser("train", help="training help")
    eval_subparser = mode_subparsers.add_parser(
        "eval", help="evaluation help", parents=[eval_parser]
    )
    json_subparser = mode_subparsers.add_parser(
        "from_json", help="load from json help", parents=[json_parser]
    )

    # train mode
    train_subparsers = train_subparser.add_subparsers(
        dest="model", help="Model-specific arguments"
    )
    # model
    schnet_subparser = train_subparsers.add_parser("schnet", help="SchNet help")
    wacsf_subparser = train_subparsers.add_parser("wacsf", help="wacsf help")

    # schnet
    schnet_subparsers = schnet_subparser.add_subparsers(
        dest="dataset", help="Dataset specific arguments"
    )
    schnet_subparsers.add_parser(
        "ani1",
        help="ANI1 dataset help",
        parents=[train_parser, schnet_parser, ani1_parser],
    )
    schnet_subparsers.add_parser(
        "matproj",
        help="Materials Project dataset help",
        parents=[train_parser, schnet_parser, matproj_parser],
    )
    schnet_subparsers.add_parser(
        "md17",
        help="MD17 dataset help",
        parents=[train_parser, schnet_parser, md17_parser],
    )
    schnet_subparsers.add_parser(
        "omdb",
        help="Organic Materials dataset help",
        parents=[train_parser, schnet_parser, omdb_parser],
    )
    schnet_subparsers.add_parser(
        "qm9",
        help="QM9 dataset help",
        parents=[train_parser, schnet_parser, qm9_parser],
    )
    schnet_subparsers.add_parser(
        "custom",
        help="Custom dataset help",
        parents=[train_parser, schnet_parser, custom_data_parser],
    )

    # wacsf
    wacsf_subparsers = wacsf_subparser.add_subparsers(
        dest="dataset", help="Dataset specific arguments"
    )
    wacsf_subparsers.add_parser(
        "ani1",
        help="ANI1 dataset help",
        parents=[train_parser, wacsf_parser, ani1_parser],
    )
    wacsf_subparsers.add_parser(
        "matproj",
        help="Materials Project dataset help",
        parents=[train_parser, wacsf_parser, matproj_parser],
    )
    wacsf_subparsers.add_parser(
        "md17",
        help="MD17 dataset help",
        parents=[train_parser, wacsf_parser, md17_parser],
    )
    wacsf_subparsers.add_parser(
        "omdb",
        help="Organic Materials dataset help",
        parents=[train_parser, wacsf_parser, omdb_parser],
    )
    wacsf_subparsers.add_parser(
        "qm9", help="QM9 dataset help", parents=[train_parser, wacsf_parser, qm9_parser]
    )
    wacsf_subparsers.add_parser(
        "custom",
        help="Custom dataset help",
        parents=[train_parser, wacsf_parser, custom_data_parser],
    )
    return mode_parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
