import torch
from torch_ema import ExponentialMovingAverage as TorchEMA

from schnetpack.lightning.callbacks import EMA


def test_foreach_update_matches_torch_ema():
    g = torch.Generator().manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(s, generator=g, dtype=torch.float64))
        for s in ((12, 7), (7,), (3, 5, 2))
    ]
    reference, ema = TorchEMA(params, decay=0.995), EMA(params, decay=0.995)
    for _ in range(50):
        with torch.no_grad():
            for p in params:
                p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        reference.update()
        ema.update()
    for a, b in zip(ema.shadow_params, reference.shadow_params):
        torch.testing.assert_close(a, b)
    assert ema.state_dict()["num_updates"] == reference.state_dict()["num_updates"]
