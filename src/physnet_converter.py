import torch
import schnetpack as spk
import re

arguments = dict(
    num_modules="n_interactions",
    num_basis_functions="n_basis_functions",
    num_features="n_atom_basis",
    num_residual_output="n_residual_post_interaction",
    num_residual_post_v="n_residuals_v",
    num_residual_post_x="n_residuals_out",
    num_residual_pre_vi="n_residuals_i",
    num_residual_pre_vj="n_residuals_j",
    num_residual_pre_x="n_residuals_in",
    basis_functions=None,
    exp_weighting=None,
    cutoff=None,
    lr_cutoff=None,
    use_zbl_repulsion=None,
    use_electrostatics=None,
    use_d4_dispersion=None,
    compute_d4_atomic=None,
    activation=None,
    module_keep_prob=None,
    load_from=None,
    Zmax7=None,
)


spk_args = dict(
    n_interactions="num_modules",
    n_basis_functions="num_basis_functions",
    n_atom_basis="num_features",
    n_residual_post_interaction="num_residual_output",
    n_residual_post_v="num_residual_post_v",
    n_residual_post_x="num_residual_post_x",
    n_residual_pre_vi="num_residual_pre_vi",
    n_residual_pre_vj="num_residual_pre_vj",
    n_residual_pre_x="num_residual_pre_x",
    exp_weighting="exp_weighting",
    cutoff="cutoff",
    max_z="Zmax",
)


def get_distance_expansion(distance_expansion, n_basis_functions, exp_weighting):
    """
    Build distance expansion layer.

    """
    if distance_expansion == "exp-bernstein":
        return spk.nn.ExponentialBernsteinPolynomials(
            num_basis_functions=n_basis_functions, exp_weighting=exp_weighting
        )
    else:
        raise NotImplementedError


def transform_residual_stack_sd(res_stack, orig_stack_sd, prefix=""):
    """
    Transform original state dict of residual stack block to spk version.

    """
    spk_stack_sd = {}
    for i, res_block in enumerate(res_stack.stack):
        spk_res_sd = {
            "{}stack.{}.layers.0.weight".format(prefix, i): orig_stack_sd[
                "stack.{}.linear1.weight".format(i)
            ],
            "{}stack.{}.layers.0.bias".format(prefix, i): orig_stack_sd[
                "stack.{}" ".linear1.bias".format(i)
            ],
            "{}stack.{}.layers.0.pre_activation.alpha".format(prefix, i): orig_stack_sd[
                "stack.{}.activation_pre.alpha".format(i)
            ],
            "{}stack.{}.layers.0.pre_activation.beta".format(prefix, i): orig_stack_sd[
                "stack.{}.activation_pre.beta".format(i)
            ],
            "{}stack.{}.layers.1.weight".format(prefix, i): orig_stack_sd[
                "stack.{}.linear2.weight".format(i)
            ],
            "{}stack.{}.layers.1.bias".format(prefix, i): orig_stack_sd[
                "stack.{}.linear2.bias".format(i)
            ],
            "{}stack.{}.layers.1.pre_activation.alpha".format(prefix, i): orig_stack_sd[
                "stack.{}.activation_post.alpha".format(i)
            ],
            "{}stack.{}.layers.1.pre_activation.beta".format(prefix, i): orig_stack_sd[
                "stack.{}.activation_post.beta".format(i)
            ],
        }
        spk_stack_sd.update(spk_res_sd)

    return spk_stack_sd


def get_sub_dict(main_dict, start_str):
    sub_dict = dict(
        (re.sub(start_str, "", k), v)
        for k, v in main_dict.items()
        if k.startswith(start_str)
    )
    return sub_dict


if __name__ == "__main__":
    # load arguments and state_dict from original model
    model_path = "stefaan_physnet/parameters.pth"
    orig_args = torch.load(model_path, map_location="cpu")
    orig_state_dict = orig_args["state_dict"]

    # build new representation block
    n_atom_basis = orig_args["num_features"]
    n_basis_functions = orig_args["num_basis_functions"]
    n_interactions = orig_args["num_modules"]
    n_residual_pre_x = orig_args["num_residual_pre_x"]
    n_residual_post_x = orig_args["num_residual_post_x"]
    n_residual_pre_vi = orig_args["num_residual_pre_vi"]
    n_residual_pre_vj = orig_args["num_residual_pre_vj"]
    n_residual_post_v = orig_args["num_residual_post_v"]
    n_residual_post_interaction = orig_args["num_residual_output"]
    distance_expansion = get_distance_expansion(
        distance_expansion=orig_args["basis_functions"],
        n_basis_functions=n_basis_functions,
        exp_weighting=orig_args["exp_weighting"],
    )
    cutoff = orig_args["cutoff"]
    # activation=spk.nn.Swish,
    max_z = orig_args["Zmax"]

    repr_args = dict(
        n_atom_basis=n_atom_basis,
        n_basis_functions=n_basis_functions,
        n_interactions=n_interactions,
        n_residual_pre_x=n_residual_pre_x,
        n_residual_post_x=n_residual_post_x,
        n_residual_pre_vi=n_residual_pre_vi,
        n_residual_pre_vj=n_residual_pre_vj,
        n_residual_post_v=n_residual_post_v,
        n_residual_post_interaction=n_residual_post_interaction,
        cutoff=cutoff,
        max_z=max_z,
        distance_expansion=distance_expansion,
        coupled_interactions=False,
        return_intermediate=False,
        return_distances=True,
        interaction_aggregation="sum",
        # cutoff_network=spk.nn.MollifierCutoff,
    )
    spk_repr = spk.PhysNet(**repr_args)

    # update embedding
    embedding_sd = dict(
        (re.sub("embedding.", "", k), orig_state_dict[k])
        for k in (
            "embedding.element_embedding",
            "embedding.electron_config",
            "embedding.config_linear.weight",
        )
    )
    spk_repr.embedding.load_state_dict(embedding_sd)

    # update basis functions
    basis_functions_keys = [
        k for k in orig_state_dict.keys() if k.startswith("radial_basis_functions")
    ]
    basis_functions_sd = dict(
        (re.sub("radial_basis_functions.", "", k), orig_state_dict[k])
        for k in basis_functions_keys
    )
    spk_repr.distance_expansion.load_state_dict(basis_functions_sd)

    # update interaction blocks by collecting a state dict for every (post-)interaction
    # block and applying it at the end of the iteration
    for i, (interaction, post_interaction) in enumerate(
        zip(spk_repr.interactions, spk_repr.post_interactions)
    ):
        # build new states dicts
        spk_interaction_sd = {}
        spk_post_interaction_sd = {}

        # original state_dict of modular block
        orig_module_sd = get_sub_dict(orig_state_dict, "module.{}.".format(i))

        # collect parameters of interaction block
        # spin and charge features
        spin_charge_sd = dict(
            charge_embedding=orig_module_sd["qfeatures"],
            charge_keys=orig_module_sd["qkey"],
            spin_features=orig_module_sd["sfeatures"],
            spin_keys=orig_module_sd["skey"],
        )

        # input residual
        orig_res_sd = get_sub_dict(orig_module_sd, "residual_pre_x.".format(i))
        input_residual_sd = transform_residual_stack_sd(
            interaction.input_residual, orig_res_sd, prefix="input_residual."
        )

        # i and j branches
        orig_res_sd = get_sub_dict(orig_module_sd, "residual_pre_vi.".format(i))
        branch_i = transform_residual_stack_sd(
            interaction.branch_i[0], orig_res_sd, prefix="branch_i.0."
        )

        def transform_dense_layer(spk_layer, orig_sd, prefix=""):
            spk_dense_sd = dict()
            spk_dense_sd["{}weight".format(prefix)] = orig_sd["weight"]
            spk_dense_sd["{}bias".format(prefix)] = orig_sd["bias"]
            if type(spk_layer.pre_activetion) == spk.nn.Swish:
                pass

        # convolution layer

        # v branch

        # output residual

        b = "reak"
        # [
        #    k for k in orig_state_dict.keys() if k.startswith("module.{}".format(i))
        # ]
        # build interaction state_dict

        # build post-interaction state_dict

    b = "reak"
