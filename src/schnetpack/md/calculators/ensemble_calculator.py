import torch

__all__ = ["EnsembleCalculator"]


class EnsembleCalculator:
    def calculate(self, system):
        inputs = self._generate_input(system)

        results = []
        for model in self.models:
            prediction = model(inputs)
            results.append(prediction)

        # Compute statistics
        self.results = self._accumulate_results(results)
        self._update_system(system)

    def _accumulate_results(self, results):
        # Get the keys
        accumulated = {p: [] for p in results[0]}

        ensemble_results = {p: [] for p in results[0]}
        for p in accumulated:
            tmp = torch.stack([result[p].detach() for result in results])
            ensemble_results[p] = torch.mean(tmp, dim=0)
            ensemble_results["{:s}_stddev".format(p)] = torch.std(tmp, dim=0)

        return ensemble_results

    def _update_required_properties(self, required_properties):
        newp = []
        for p in required_properties:
            newp += [p, "{:s}_stddev".format(p)]
        self.required_properties = newp
