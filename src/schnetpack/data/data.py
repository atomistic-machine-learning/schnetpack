from typing import (Any, Dict, Sequence, Tuple)
from dataclasses import dataclass
Array = Any
DataTupleT = Tuple[Dict[str, Array], Dict[str, Any]]


@dataclass
class DataTuple:
    inputs: Sequence[str]
    targets: Sequence[str]
    prop_keys: Dict[str, str]

    def __post_init__(self):
        self.input_keys = [self.prop_keys[i] for i in self.inputs]
        self.target_keys = [self.prop_keys[t] for t in self.targets]
        self.get_args = lambda data, args: {k: v for (k, v) in data.items() if k in args}

    def __call__(self, ds: Dict[str, Array]) -> DataTupleT:
        inputs = self.get_args(ds, self.input_keys)
        targets = self.get_args(ds, self.target_keys)
        return inputs, targets