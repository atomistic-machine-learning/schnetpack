#!/usr/bin/env python
import os
import logging
import argparse
from schnetpack.data import generate_db


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to xyz-file or extxyz-file with molecular data.",
    )
    parser.add_argument("db_path", type=str, help="Path to output database.")
    parser.add_argument(
        "--atomic_properties",
        type=str,
        help="String with definition of atomic properties (e.g. forces) contained in "
        "input file. Only needed for .xyz-files. (default: %(default)s)",
        default="Properties=species:S:1:pos:R:3",
    )
    parser.add_argument(
        "--molecular_properties",
        type=str,
        help="Molecular properties (e.g. energy, homo/lumo, ...) contained in data "
        "file. Only needed for xyz-files. (default: %(default)s)",
        nargs="+",
        default=["energy"],
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing database file."
    )

    return parser


def main(args):
    if args.overwrite and os.path.exists(args.db_path):
        logging.info("Removing old database at {}".format(args.db_path))
        os.remove(args.db_path)

    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    logging.info("parsing data...")
    generate_db(
        args.file_path, args.db_path, args.atomic_properties, args.molecular_properties
    )

    logging.info("done...")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
