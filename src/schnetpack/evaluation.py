import numpy as np
from schnetpack.data.definitions import Structure


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
            dataloader (torch.utils.Dataloader): Dataloader with data to
                evaluate
            device (str): cpu or cuda

        Returns:

        """
        predicted = {}
        for batch in self.dataloader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = self.model(batch)

            for p in result.keys():
                value = result[p].cpu().detach().numpy()
                if p in predicted.keys():
                    predicted[p].append(value)
                else:
                    predicted[p] = [value]

            for p in [Structure.R, Structure.Z]:
                value = batch[p].cpu().detach().numpy()
                if p in predicted.keys():
                    predicted[p].append(value)
                else:
                    predicted[p] = [value]

        for p in predicted.keys():
            predicted[p] = np.vstack(predicted[p])

        return predicted

    def evaluate(self, device):
        raise NotImplementedError


class XYZEvaluator(Evaluator):

    def __init__(self, model, dataloader, out_file):
        self.out_file = out_file
        super(XYZEvaluator, self).__init__(model, dataloader)

    def evaluate(self, device):
        predicted = self._get_predicted(device)
        np.savez(self.out_file, **predicted)
