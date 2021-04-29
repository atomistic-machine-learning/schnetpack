import os
import torch
import numpy as np


def convert(old_statedict):
    print(schnet_new.state_dict().keys())
    print()
    print(schnet_old.state_dict().keys())

    names = [
        ("embedding.weight", "embedding.weight"),
        ("radial_basis.offsets", "distance_expansion.offsets"),
        ("radial_basis.widths", "distance_expansion.width"),
        (
            "interactions.0.filter_network.0.weight",
            "interactions.0.cfconv.filter_network.0.weight",
        ),
        (
            "interactions.0.filter_network.0.bias",
            "interactions.0.cfconv.filter_network.0.bias",
        ),
        (
            "interactions.0.filter_network.1.weight",
            "interactions.0.cfconv.filter_network.1.weight",
        ),
        (
            "interactions.0.filter_network.1.bias",
            "interactions.0.cfconv.filter_network.1.bias",
        ),
        ("interactions.0.in2f.weight", "interactions.0.cfconv.in2f.weight"),
        ("interactions.0.f2out.0.weight", "interactions.0.cfconv.f2out.weight"),
        ("interactions.0.f2out.0.bias", "interactions.0.cfconv.f2out.bias"),
        ("interactions.0.f2out.1.weight", "interactions.0.dense.weight"),
        ("interactions.0.f2out.1.bias", "interactions.0.dense.bias"),
        ("cutoff_fn.cutoff", "interactions.0.cutoff_network.cutoff"),
        ("cutoff_fn.cutoff", "interactions.0.cfconv.cutoff_network.cutoff"),
        (
            "interactions.1.filter_network.0.weight",
            "interactions.1.cfconv.filter_network.0.weight",
        ),
        (
            "interactions.1.filter_network.0.bias",
            "interactions.1.cfconv.filter_network.0.bias",
        ),
        (
            "interactions.1.filter_network.1.weight",
            "interactions.1.cfconv.filter_network.1.weight",
        ),
        (
            "interactions.1.filter_network.1.bias",
            "interactions.1.cfconv.filter_network.1.bias",
        ),
        ("interactions.1.in2f.weight", "interactions.1.cfconv.in2f.weight"),
        ("interactions.1.f2out.0.weight", "interactions.1.cfconv.f2out.weight"),
        ("interactions.1.f2out.0.bias", "interactions.1.cfconv.f2out.bias"),
        ("interactions.1.f2out.1.weight", "interactions.1.dense.weight"),
        ("interactions.1.f2out.1.bias", "interactions.1.dense.bias"),
    ]

    nstate = schnet_new.state_dict()
    ostate = schnet_old.state_dict()

    for nname, oname in names:
        # if nname != "cutoff_fn.cutoff":
        #     ostate[oname].copy_(torch.rand_like(ostate[oname]))
        nstate[nname].copy_(ostate[oname])
        print(nname, oname, ostate[oname], nstate[nname])
        assert np.allclose(nstate[nname].cpu(), ostate[oname])
