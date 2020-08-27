import schnetpack as spk
import torch.nn as nn


__all__ = ["AtomisticModelUpdate"]


class UpdateError(Exception):
    pass


class SchNetPackUpdate:
    def __init__(self):
        self.newest_version = spk.__version__


class AtomisticModelUpdate(SchNetPackUpdate):
    def __init__(self, model):
        super(AtomisticModelUpdate, self).__init__()
        self.model = model

    @property
    def _model_version(self):
        if hasattr(self.model, "version"):
            return self.model.version
        return "<=0.3.1"

    def update_to(self, target_version):
        # update representation
        representation_updater = RepresentationUpdate(self.model.representation)
        representation_updater.update_to(target_version=target_version)

        # update output modules
        for output_module in self.model.output_modules:
            output_module_updater = OutputModuleUpdate(output_module=output_module)
            output_module_updater.update_to(target_version=target_version)

        # from <= 0.3.1 to 0.3.2
        if self._model_version == "<=0.3.1":
            if target_version != "<=0.3.1":

                # set requires_stress attribute
                if not hasattr(self.model, "requires_stress"):
                    self.model.requires_stress = any(
                        [om.stress for om in self.model.output_modules]
                    )

                # set version tag
                self.model.version = "0.3.2"

    def update(self):
        self.update_to(self.newest_version)


class RepresentationUpdate(SchNetPackUpdate):
    def __init__(self, model):
        super(RepresentationUpdate, self).__init__()
        self.model = model

    @property
    def _model_version(self):
        if hasattr(self.model, "version"):
            return self.model.version
        return "<=0.3.1"

    def update_schnet_to(self, target_version=None):
        if target_version is None:
            target_version = self.newest_version

        # from <= 0.3.1 to 0.3.1.nightly
        if self._model_version == "<=0.3.1":
            if target_version != "<=0.3.1":
                # rename distance provider
                self.model.distance_provider = self.model.distances
                del self.model.distances

                # add pre_interactions
                self.model.pre_interactions = nn.ModuleList(
                    nn.Identity() for _ in range(len(self.model.interactions))
                )

                # interaction refinements
                self.model.interaction_refinements = nn.ModuleList(
                    spk.nn.FeatureSum([]) for _ in range(len(self.model.interactions))
                )

                # add post_interactions
                self.model.post_interactions = nn.ModuleList(
                    nn.Identity() for _ in range(len(self.model.interactions))
                )

                # interaction outputs
                self.model.interaction_outputs = nn.ModuleList(
                    nn.Identity() for _ in range(len(self.model.interactions))
                )

                # add interaction aggregation
                self.model.interaction_aggregation = spk.representation.InteractionAggregation(
                    "last"
                )

                # add return_distances argument
                self.model.return_distances = False

                # add pre_activation to Dense layers
                for module in self.model.modules():
                    if type(module) == spk.nn.Dense:
                        module.pre_activation = nn.Identity()
                        if module.activation is None:
                            module.activation = nn.Identity()

                # set version tag
                self.model.version = "0.3.2"

    def update_physnet_to(self, target_version=None):
        if target_version is None:
            target_version = self.newest_version

        # since physnet is introduced with spk 0.3.2, nothing needs to be updated yet

    def update_to(self, target_version):
        if type(self.model) == spk.SchNet:
            self.update_schnet_to(target_version=target_version)
        elif type(self.model) == spk.PhysNet:
            self.update_physnet_to(target_version=target_version)
        else:
            pass


class OutputModuleUpdate(SchNetPackUpdate):
    def __init__(self, output_module):
        super(OutputModuleUpdate, self).__init__()
        self.output_module = output_module

    @property
    def _model_version(self):
        if hasattr(self.output_module, "version"):
            return self.output_module.version
        return "<=0.3.1"

    def update_to(self, target_version=None):
        # get target version
        if target_version is None:
            target_version = self.newest_version

        # from <= 0.3.1 to 0.3.2
        if self._model_version == "<=0.3.1":
            if target_version != "<=0.3.1":

                # check if stress attribute exists
                if not hasattr(self.output_module, "stress"):
                    self.output_module.stress = None

                # add pre_activation to Dense layers
                for module in self.output_module.modules():
                    if type(module) == spk.nn.Dense:
                        module.pre_activation = nn.Identity()
                        if module.activation is None:
                            module.activation = nn.Identity()

                # set new version tag
                self.output_module.version = "0.3.2"


class AtomsLoaderUpdate(SchNetPackUpdate):
    def __init__(self):
        super(AtomsLoaderUpdate, self).__init__()


class AtomsDatasetUpdate(SchNetPackUpdate):
    def __init__(self):
        super(AtomsDatasetUpdate, self).__init__()
