import argparse
import logging

from schnetpack.md.parsers.md_setup import MDSimulation

try:
    import oyaml as yaml
except ImportError:
    import yaml


def read_options(yamlpath):
    with open(yamlpath, "r") as tf:
        tradoffs = yaml.load(tf)

    logging.info("Read options from {:s}.".format(yamlpath))
    return tradoffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("md_input")
    args = parser.parse_args()

    config = read_options(args.md_input)

    md = MDSimulation(config)
    md.save_config()
    md.run()
