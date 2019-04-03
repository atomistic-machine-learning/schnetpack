import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for training and prediction (default: %(default)s)",
        default=100,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to train the model on",
        default='cuda',
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="path to model_dir",
        default='training',
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to dataset",
        default='data',
    )
    return parser
