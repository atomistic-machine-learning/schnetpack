from functools import partial

from torch.nn.init import constant_

zeros_initializer = partial(constant_, val=0.0)
