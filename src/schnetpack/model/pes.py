import logging

import hydra
import pytorch_lightning.metrics
import torch

import schnetpack as spk
import cProfile

from schnetpack.model.base import AtomisticModel

log = logging.getLogger(__name__)


class PESModel(AtomisticModel):
    """
    AtomisticModel for potential energy surfaces
    """

    def build_model(
        self,
    ):
        self.representation = hydra.utils.instantiate(self._representation_cfg)

        self.loss_fn = hydra.utils.instantiate(self._output_cfg.loss)

        if self._output_cfg.requires_atomref:
            atomrefs = self.datamodule.train_dataset.atomrefs
            atomref = atomrefs[self._output_cfg.property][:, None]
        else:
            atomrefs = None
            atomref = None

        if self._output_cfg.requires_stats:
            log.info("Calculate stats...")
            stats = spk.data.calculate_stats(
                self.datamodule.train_dataloader(),
                divide_by_atoms={
                    self._output_cfg.property: self._output_cfg.divide_stats_by_atoms
                },
                atomref=atomrefs,
            )[self._output_cfg.property]
            log.info(
                f"{self._output_cfg.property} (mean / stddev): {stats[0]}, {stats[1]}"
            )

            self.output = hydra.utils.instantiate(
                self._output_cfg.module,
                atomref=atomref,
                mean=torch.tensor(stats[0], dtype=torch.float32),
                stddev=torch.tensor(stats[1], dtype=torch.float32),
            )
        else:
            atomref = atomref[:, None] if atomref else None
            self.output = hydra.utils.instantiate(
                self._output_cfg.module, atomref=atomref
            )

        self.metrics = torch.nn.ModuleDict(
            {
                name: hydra.utils.instantiate(metric)
                for name, metric in self._output_cfg.metrics.items()
            }
        )
