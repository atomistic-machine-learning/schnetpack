"""
SchNetPack's model format: a folder with ``hyperparameters.json``, the classes and
constructor arguments of a model in the shape of the Hydra ``model:`` block, and
``model.safetensors``, its weights and statistics.

Every SchNetPack module class is decorated with :func:`register`, which enters it in
the table of classes a model file may name and makes each instance record the
arguments it is built with. :func:`save_model` writes these records;
:func:`load_model` builds the model from them with its own small builder over that
table and loads the weights strictly. Loading runs no code from the files and needs
no Hydra.
"""

import functools
import inspect
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn

from schnetpack.utils import as_dtype

__all__ = ["register", "save_model", "load_model", "load_hyperparameters"]

#: the format this library writes and the newest it reads
FORMAT = 1

#: format n -> n + 1: a function that edits the hyperparameters and the weights of a
#: file of format n before the model is built (a class or argument renamed, weights
#: renamed or reshaped). Empty while there is one format.
MIGRATIONS: dict[int, Callable[[dict, dict], tuple[dict, dict]]] = {}

#: the registered classes, by the module and name they are defined under
_CLASSES: dict[str, type] = {}

#: torch modules a SchNetPack class may take as an argument (an activation, PaiNN's
#: nuclear embedding). Each is written with its ``__constants__``, which must all be
#: constructor arguments: check that before adding one.
_TORCH_MODULES = {"torch.nn.SiLU": nn.SiLU, "torch.nn.Embedding": nn.Embedding}

_HYPERPARAMETERS = "hyperparameters.json"
_WEIGHTS = "model.safetensors"


def _functions() -> dict[str, Callable]:
    """Functions a SchNetPack class may take as an argument."""
    from schnetpack.nn.activations import shifted_softplus

    return {
        "torch.nn.functional.silu": nn.functional.silu,
        "schnetpack.nn.activations.shifted_softplus": shifted_softplus,
        "torch.nn.init.xavier_uniform_": nn.init.xavier_uniform_,
        "torch.nn.init.zeros_": nn.init.zeros_,
    }


def register(cls: type[nn.Module]) -> type[nn.Module]:
    """
    Make a module class savable with :func:`save_model` and loadable with
    :func:`load_model`.

    The class is entered in the table of classes a model file may name, and each
    instance keeps the arguments it was built with. A class of your own (a head, a
    representation) needs this one line too, and must be imported before a model
    that contains it is loaded::

        @spk.register
        class MyHead(nn.Module):
            def __init__(self, n_in: int, output_key: str): ...

    Only the arguments given are kept, not the defaults. A module unpickled or
    copied keeps the record it had; one that was never built through a registered
    class has none, and cannot be saved.
    """
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise TypeError(f"register takes an nn.Module class, not {cls!r}")
    init = cls.__init__
    signature = inspect.signature(init)

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        # the class that is built records, not its registered parents
        if type(self) is cls:
            bound = signature.bind(self, *args, **kwargs).arguments
            record = {}
            for name, value in list(bound.items())[1:]:
                if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    record.update(value)
                else:
                    record[name] = value
            self.__dict__["_init_args"] = record

    cls.__init__ = __init__
    _CLASSES[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return cls


def _names(cls: type) -> list[str]:
    """The paths a class is found under, shortest first: the module that defines
    it and every package above it that exports it (``schnetpack.model.PaiNN``)."""
    names = {f"{cls.__module__}.{cls.__qualname__}"}
    parts = cls.__module__.split(".")
    for i in range(1, len(parts)):
        package = sys.modules.get(".".join(parts[:i]))
        if getattr(package, cls.__name__, None) is cls:
            names.add(f"{package.__name__}.{cls.__name__}")
    return sorted(names, key=lambda name: (len(name), name))


def _classes() -> dict[str, type]:
    """The classes a model file may name, by every path they are found under."""
    table = {name: cls for cls in _CLASSES.values() for name in _names(cls)}
    return table | _TORCH_MODULES


def _tree(value, where: str, modules: list):
    """``value`` as plain JSON: a module as ``{"_target_": its public name, **the
    arguments it was built with}``, in the shape of a Hydra config. Each module
    met is added to ``modules`` with its path."""
    if isinstance(value, nn.ModuleList):  # e.g. the models of an NNEnsemble
        return [_tree(v, f"{where}[{i}]", modules) for i, v in enumerate(value)]
    if isinstance(value, nn.Module):
        modules.append((where, value))
        cls = type(value)
        for name, torch_cls in _TORCH_MODULES.items():
            if cls is torch_cls:
                constants = {k: getattr(value, k) for k in cls.__constants__}
                return {"_target_": name, **constants}
        record = vars(value).get("_init_args")
        if record is None:
            raise ValueError(
                f"{where}: this {cls.__qualname__} has no record of its constructor "
                "arguments. Its class must be decorated with @schnetpack.register "
                "and the module built through it; a module unpickled from a file "
                "written without the record has none."
            )
        node = {"_target_": _names(cls)[0]}
        node.update({k: _tree(v, f"{where}.{k}", modules) for k, v in record.items()})
        if isinstance(vars(value).get("do_postprocessing"), bool):
            # a switch the library sets on built models (training, MD, saving): it
            # is saved as it is now, not as it was built
            node["do_postprocessing"] = value.do_postprocessing
        return node
    if isinstance(value, torch.Tensor):
        # the class keeps it as a buffer or parameter: the values come from the weights
        dtype = str(value.dtype).removeprefix("torch.")
        return {"_tensor_": {"shape": list(value.shape), "dtype": dtype}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return {k: _tree(v, f"{where}.{k}", modules) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_tree(v, f"{where}[{i}]", modules) for i, v in enumerate(value)]
    if isinstance(value, functools.partial) and not value.args and not value.keywords:
        value = value.func  # what Hydra's {_target_: ..., _partial_: true} gives
    for name, function in _functions().items():
        if value is function:
            return {"_target_": name, "_partial_": True}
    raise ValueError(
        f"{where}: cannot write {value!r:.80} ({type(value).__name__}). A model file "
        "holds JSON values, registered modules, tensors kept as weights, "
        f"{', '.join(_TORCH_MODULES)} and the functions {', '.join(_functions())}."
    )


def _build(node, classes: dict[str, type]):
    """The object a node of ``hyperparameters.json`` describes, children first."""
    if isinstance(node, list):
        return [_build(v, classes) for v in node]
    if not isinstance(node, dict):
        return node
    if "_tensor_" in node:
        dtype = as_dtype(node["_tensor_"]["dtype"])
        return torch.zeros(node["_tensor_"]["shape"], dtype=dtype)
    if "_target_" not in node:
        return {k: _build(v, classes) for k, v in node.items()}
    name = node["_target_"]
    if "_partial_" in node:
        functions = _functions()
        if (
            name not in functions
            or node["_partial_"] is not True
            or set(node) != {"_target_", "_partial_"}
        ):
            raise ValueError(f"{node} is not a function a model file may name")
        return functions[name]
    if name not in classes:
        raise ValueError(
            f"{name!r} is not a class a model file may name. The table holds the "
            f"registered classes (@schnetpack.register) and {', '.join(_TORCH_MODULES)}"
            "; a class of your own must be imported before the model is loaded."
        )
    kwargs = {k: _build(v, classes) for k, v in node.items() if k != "_target_"}
    return classes[name](**kwargs)


def _rebuild(tree: dict) -> nn.Module:
    """The model ``tree`` describes, with fresh weights; the global random state
    is left as it was."""
    with torch.random.fork_rng(devices=[]):
        return _build(tree, _classes())


def _check_strings(node) -> None:
    """Refuse OmegaConf's ``${...}``: a model file is plain JSON."""
    if isinstance(node, dict):
        for key, value in node.items():
            _check_strings(key)
            _check_strings(value)
    elif isinstance(node, list):
        for value in node:
            _check_strings(value)
    elif isinstance(node, str) and "${" in node:
        raise ValueError(f"{node!r}: a model file holds no ${{...}} interpolations")


def _differences(model: nn.Module, rebuilt: nn.Module) -> list[str]:
    """What the module tree and the weights' names and shapes of ``model`` have
    that ``rebuilt`` lacks, and the reverse."""

    def layout(m):
        modules = {
            f"{n or '<model>'}: {type(x).__qualname__}" for n, x in m.named_modules()
        }
        weights = {f"{k}: {tuple(v.shape)}" for k, v in m.state_dict().items()}
        return modules | weights

    live, built = layout(model), layout(rebuilt)
    return [f"only in the model {x}" for x in sorted(live - built)] + [
        f"only when built again {x}" for x in sorted(built - live)
    ]


def save_model(model: nn.Module, path: str, blocks: dict | None = None) -> None:
    """
    Save a model to the folder ``path``: ``hyperparameters.json`` with its classes
    and constructor arguments, and ``model.safetensors`` with its weights.

    The arguments are those the model was built with (see :func:`register`). Before
    anything is written, the model is built again from them and compared with
    ``model``. If a module was added, removed or replaced after construction, or a
    weight has another name or shape, the file would load as another model, and
    nothing is written: construct the model the way it should be saved instead.
    A module replaced by another, even of the same class and shapes, is refused too.

    Args:
        model: a model built from registered classes.
        path: the model folder; created if missing, its two files replaced.
        blocks: further plain-JSON entries of ``hyperparameters.json`` next to
            ``model``, e.g. a generative model's ``generative`` and ``sample``
            settings; read back with :func:`load_hyperparameters`.
    """
    import schnetpack

    if os.path.exists(path) and not os.path.isdir(path):
        raise FileExistsError(
            f"{path} is a file, most likely a model pickled by an earlier SchNetPack "
            "version; a model is now saved as a folder. Move the file away or save "
            "to another path."
        )
    blocks = blocks or {}
    reserved = {"format", "schnetpack_version", "model"} & set(blocks)
    if reserved:
        raise ValueError(f"blocks may not be named {sorted(reserved)}")
    modules = []  # every module the records refer to, with its path
    hyperparameters = {
        "format": FORMAT,
        "schnetpack_version": schnetpack.__version__,
        "model": _tree(model, "model", modules),
        **blocks,
    }
    live = {id(m) for m in model.modules()}
    replaced = [where for where, m in modules if id(m) not in live]
    if replaced:
        raise ValueError(
            f"the model was changed after it was built: {', '.join(replaced)} is no "
            "longer the module it was built with, so the file would describe the old "
            "one. Nothing was saved: construct the model with that module, then save it."
        )
    try:
        text = json.dumps(hyperparameters, indent=2)
    except (TypeError, ValueError) as err:
        raise ValueError(f"the blocks are not plain JSON: {err}") from err
    plain = json.loads(text)  # as a reader of the file gets it
    _check_strings(plain)

    try:
        rebuilt = _rebuild(plain["model"])
    except Exception as err:
        raise ValueError(
            f"the model cannot be built again from its constructor arguments: {err}"
        ) from err
    differences = _differences(model, rebuilt)
    if differences:
        raise ValueError(
            "the model was changed after it was built, so its constructor arguments "
            "build another model (" + "; ".join(differences[:6]) + "). Nothing was "
            "saved: construct the model the way it should be saved, then save it."
        )

    from safetensors.torch import save

    os.makedirs(path, exist_ok=True)
    _replace(os.path.join(path, _HYPERPARAMETERS), (text + "\n").encode())
    # copies: safetensors stores no tensors that share memory, as the weights of a
    # module repeated in a list (shared_interactions) do
    state = {
        k: v.detach().to("cpu", memory_format=torch.contiguous_format, copy=True)
        for k, v in model.state_dict().items()
    }
    # the bytes are written here, not by safetensors' save_file, which makes the
    # file readable by its owner only (0600) whatever the umask
    _replace(os.path.join(path, _WEIGHTS), save(state))


def _replace(file: str, data: bytes) -> None:
    """Write ``file`` whole or not at all: training rewrites best_model while
    others may read it, and a crash must not leave half a file."""
    temporary = f"{file}.{os.getpid()}.tmp"
    with open(temporary, "wb") as f:
        f.write(data)
    os.replace(temporary, file)


def _read(path: str, weights: bool) -> tuple[dict, dict]:
    """The hyperparameters and (if ``weights``) the weights of a model folder,
    checked and migrated to the current format."""
    with open(os.path.join(path, _HYPERPARAMETERS)) as f:
        hyperparameters = json.load(f)
    version = (
        hyperparameters.get("format") if isinstance(hyperparameters, dict) else None
    )
    if not isinstance(version, int) or version < 1:
        raise ValueError(
            f"{path} is not a SchNetPack model folder (format {version!r})"
        )
    if version > FORMAT:
        raise ValueError(
            f"{path} is in model format {version}, and this SchNetPack reads up to "
            f"format {FORMAT}: upgrade SchNetPack to load it."
        )
    _check_strings(hyperparameters)
    state = {}
    if weights:
        from safetensors.torch import load_file

        state = load_file(os.path.join(path, _WEIGHTS))
    for old in range(version, FORMAT):
        hyperparameters, state = MIGRATIONS[old](hyperparameters, state)
    return hyperparameters, state


def load_hyperparameters(path: str) -> dict:
    """
    ``hyperparameters.json`` of the model folder ``path``, in the current format:
    ``model`` and the blocks saved with it (e.g. ``generative``, ``sample``).
    """
    return _read(path, weights=False)[0]


def load_model(path: str, device: str | torch.device = "cpu") -> nn.Module:
    """
    Load a model saved with :func:`save_model`, in eval mode.

    The classes are looked up in the table of registered classes and the weights
    are loaded strictly, each with the dtype it was saved with. Nothing from the
    files is run, and Hydra is not needed.

    A ``path`` that is a file, not a folder, is a pickled model (SchNetPack 2.x, or
    3.x before this format) and is loaded by `schnetpack.utils.load_model` as
    before. Unpickling runs code from the file: load only files you trust.

    Args:
        path: the model folder.
        device: the device the model is put on.
    """
    if not os.path.isdir(path):
        from schnetpack.utils.compatibility import load_model as load_pickle

        return load_pickle(path, device=device)
    hyperparameters, state = _read(path, weights=True)
    model = _rebuild(hyperparameters["model"])
    # assign: each tensor keeps its saved dtype (float64 statistics in a float32 model)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device).eval()
