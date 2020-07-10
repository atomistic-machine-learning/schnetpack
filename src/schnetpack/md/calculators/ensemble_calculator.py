import torch
import schnetpack

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
            ensemble_results["{:s}_var".format(p)] = torch.var(tmp, dim=0)

        return ensemble_results

    def _activate_stress(self, stress_handle):
        for model in self.models:
            schnetpack.utils.activate_stress_computation(model, stress=stress_handle)

    @staticmethod
    def _update_required_properties(required_properties):
        new_required = []
        for p in required_properties:
            prop_string = "{:s}_var".format(p)
            new_required += [p, prop_string]
            # Update properties
            # self.required_properties += [prop_string]
            # Update conversion
            # self.property_conversion[prop_string] = self.property_conversion[p]
        return new_required
