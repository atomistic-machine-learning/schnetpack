import torch

__all__ = ["YSWeights"]


class YSWeights:
    """
    Weights for Yoshida-Suzuki integration used in propagating the Nose-Hoover chain thermostats.

    Args:
        device (str): Device used for computation (default='cuda').
    """

    YS_weights = {
        3: torch.tensor(
            [1.35120719195966, -1.70241438391932, 1.35120719195966], dtype=torch.float64
        ),
        5: torch.tensor(
            [
                0.41449077179438,
                0.41449077179438,
                -0.65796308717750,
                0.41449077179438,
                0.41449077179438,
            ],
            dtype=torch.float64,
        ),
        7: torch.tensor(
            [
                0.78451361047756,
                0.23557321335936,
                -1.17767998417887,
                1.31518632068390,
                -1.17767998417887,
                0.23557321335936,
                0.78451361047756,
            ],
            dtype=torch.float64,
        ),
    }

    def get_weights(self, order):
        """
        Get the weights required for an integration scheme of the desired order.

        Args:
            order (int): Desired order of the integration scheme.

        Returns:
            torch.tensor: Tensor of the integration weights
        """
        if order not in self.YS_weights:
            raise ValueError(
                "Order {:d} not supported for YS integration weights".format(order)
            )
        else:
            return self.YS_weights[order]
