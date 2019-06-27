#!/usr/bin/env python
import os
import logging
from schnetpack.data import generate_db
from schnetpack.utils import get_parsing_parser


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


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
    parser = get_parsing_parser()
    args = parser.parse_args()
    main(args)
