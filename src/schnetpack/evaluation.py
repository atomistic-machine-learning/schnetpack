import numpy as np
from ase.db import connect
from ase import Atoms
from schnetpack import Properties


class Evaluator:
    def __init__(self, model, dataloader):
        """
        Base class for model predictions.

        Args:
            model (torch.nn.Module): trained model
        """
        self.model = model
        self.dataloader = dataloader

    def _get_predicted(self, device):
        """
        Calculate the predictions for the dataloader.

        Args:
            device (str): cpu or cuda

        Returns:

        """
        predicted = {}
        for batch in self.dataloader:
            # build batch for prediction
            batch = {k: v.to(device) for k, v in batch.items()}
            # predict
            result = self.model(batch)
            # store prediction batches to dict
            for p in result.keys():
                value = result[p].cpu().detach().numpy()
                if p in predicted.keys():
                    predicted[p].append(value)
                else:
                    predicted[p] = [value]
            # store positions, numbers and mask to dict
            for p in [Properties.R, Properties.Z, Properties.atom_mask]:
                value = batch[p].cpu().detach().numpy()
                if p in predicted.keys():
                    predicted[p].append(value)
                else:
                    predicted[p] = [value]

        max_shapes = {
            prop: max([list(val.shape) for val in values])
            for prop, values in predicted.items()
        }
        for prop, values in predicted.items():
            max_shape = max_shapes[prop]
            predicted[prop] = np.vstack(
                [
                    np.lib.pad(
                        batch,
                        [
                            [0, add_dims]
                            for add_dims in max_shape - np.array(batch.shape)
                        ],
                        mode="constant",
                    )
                    for batch in values
                ]
            )

        return predicted

    def evaluate(self, device):
        raise NotImplementedError


class NPZEvaluator(Evaluator):
    def __init__(self, model, dataloader, out_file):
        self.out_file = out_file
        super(NPZEvaluator, self).__init__(model=model, dataloader=dataloader)

    def evaluate(self, device):
        predicted = self._get_predicted(device)
        np.savez(self.out_file, **predicted)


class DBEvaluator(Evaluator):
    def __init__(self, model, dataloader, out_file):
        self.dbpath = dataloader.dataset.dbpath
        self.out_file = out_file
        super(DBEvaluator, self).__init__(model=model, dataloader=dataloader)

    def evaluate(self, device):
        predicted = self._get_predicted(device)
        positions = predicted.pop(Properties.R)
        atomic_numbers = predicted.pop(Properties.Z)
        atom_masks = predicted.pop(Properties.atom_mask).astype(bool)
        with connect(self.out_file) as conn:
            for i, mask in enumerate(atom_masks):
                z = atomic_numbers[i, mask]
                r = positions[i, mask]
                ats = Atoms(numbers=z, positions=r)
                data = {
                    prop: self._unpad(mask, values[i])
                    for prop, values in predicted.items()
                }
                conn.write(ats, data=data)

    def _unpad(self, mask, values):
        if len(values.shape) == 1:
            return values
        return values[mask]
