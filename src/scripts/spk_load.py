import argparse
import logging
import os

import schnetpack.datasets as dset

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    data_subparsers = main_parser.add_subparsers(dest="dataset", help="Select dataset")
    data_subparsers.required = True
    qm9_parser = data_subparsers.add_parser("qm9", help="QM9 dataset")
    qm9_parser.add_argument("dbpath", help="Destination path")
    qm9_parser.add_argument(
        "--remove_uncharacterized",
        help="Remove uncharacterized molecules from QM9",
        type=bool,
        default="False",
    )

    md17_parser = data_subparsers.add_parser("md17", help="MD17 datasets")
    md17_parser.add_argument(
        "molecule", help="Molecule dataset", choices=dset.MD17.existing_datasets
    )
    md17_parser.add_argument("dbpath", help="Destination path")

    args = main_parser.parse_args()

    if args.dataset == "qm9":
        dset.QM9(args.dbpath, True)
    if args.dataset == "md17":
        dset.MD17(args.dbpath, args.molecule)
    else:
        print("Unknown dataset!")
