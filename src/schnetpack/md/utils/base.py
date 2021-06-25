import logging

log = logging.getLogger(__name__)


def set_random_seed(seed):
    """
    This function sets the random seed (if given) or creates one for torch and numpy random state initialization

    Args:
        seed (int, optional): if seed not present, it is generated based on time
    """
    import time
    import numpy as np
    import torch

    # 1) if seed not present, generate based on time
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
            ((seed & 0xFF000000) >> 24)
            + ((seed & 0x00FF0000) >> 8)
            + ((seed & 0x0000FF00) << 8)
            + ((seed & 0x000000FF) << 24)
        )
    # 2) Set seed for numpy (e.g. splitting)
    np.random.seed(seed)
    # 3) Set seed for torch (manual_seed now seeds all CUDA devices automatically)
    torch.manual_seed(seed)
    log.info("Random state initialized with seed {:<10d}".format(seed))
