import logging

import hydra
import schnetpack

import schnetpack.transform.neighborlist

from schnetpack.model import PESModel

log = logging.getLogger(__name__)


class PESModelSF(PESModel):
    """
    AtomisticModel for potential energy surfaces using symmetry functions.
    """

    def build_model(self, datamodule: schnetpack.data.AtomsDataModule):

        self.representation = hydra.utils.instantiate(self._representation_cfg)

        # Perform SF specific updates to transforms and access data module for normalization
        if isinstance(
            self.representation, schnetpack.representation.symfuncs.SymmetryFunctions
        ):
            # Turn on collection of atom triples if representation requires angles
            if self.representation.n_basis_angular > 0:
                log.info("Enabling collection of atom triples for angular functions...")
                datamodule.train_transforms.append(
                    schnetpack.transform.neighborlist.CollectAtomTriples()
                )
                datamodule.val_transforms.append(
                    schnetpack.transform.neighborlist.CollectAtomTriples()
                )
                datamodule.test_transforms.append(
                    schnetpack.transform.neighborlist.CollectAtomTriples()
                )

            # Standardize symmetry functions
            log.info("Standardizing symmetry functions...")
            self.representation.standardize(datamodule.train_dataloader())

        self.props = {}
        if "energy" in self._output_cfg:
            self.props["energy"] = self._output_cfg.energy.property

        self.predict_forces = "forces" in self._output_cfg
        if self.predict_forces:
            self.props["forces"] = self._output_cfg.forces.property

        self.predict_stress = "stress" in self._output_cfg
        if self.predict_stress:
            self.props["stress"] = self._output_cfg.stress.property

        # Determine shape of the representation
        log.info(
            "Overall representation length: {:d}".format(
                self.representation.n_atom_basis
            )
        )
        self._output_cfg.module["n_in"] = self.representation.n_atom_basis
        # Also set shape for custom outnets
        if "custom_outnet" in self._output_cfg.module:
            if self._output_cfg.module.custom_outnet is not None:
                self._output_cfg.module.custom_outnet[
                    "n_in"
                ] = self.representation.n_atom_basis

        self.output = hydra.utils.instantiate(self._output_cfg.module)

        self.losses = {}
        for prop in ["energy", "forces", "stress"]:
            try:
                loss_fn = hydra.utils.instantiate(self._output_cfg[prop].loss)
                loss_weight = self._output_cfg[prop].loss_weight
                self.losses[prop] = (loss_fn, loss_weight)
            except Exception as e:
                print(e)

        self._collect_metrics()
