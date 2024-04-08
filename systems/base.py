import torch
import pytorch_lightning as pl

import models
from tqdm.auto import tqdm
from systems.utils import parse_optimizer, parse_scheduler, update_module_step
from utils.mixins import SaverMixin
from utils.misc import config_to_primitive, get_rank


class BaseSystem(pl.LightningModule, SaverMixin):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare()
        self.model = models.make(self.config.model.name, self.config.model)

    def prepare(self):
        pass

    def reinit_occupancy_grid(self):
        pass

    def forward(self, batch):
        raise NotImplementedError

    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            assert len(value) == 4 or len(value) == 3
            if len(value) == 3:
                # value = [0] + value
                start_value, end_value, end_step = value
                if isinstance(end_step, int):
                    current_step = self.global_step
                    value = start_value if current_step < end_step else end_value
                elif isinstance(end_step, float):
                    current_step = self.current_epoch
                    value = start_value if current_step < end_step else end_value
            elif len(value) == 4:
                start_step, start_value, end_value, end_step = value
                if isinstance(end_step, int):
                    current_step = self.global_step
                    value = (
                        (
                            start_value
                            + (end_value - start_value)
                            * max(
                                min(
                                    1.0,
                                    (current_step - start_step)
                                    / (end_step - start_step),
                                ),
                                0.0,
                            )
                        )
                        if current_step >= start_step
                        else 0.0
                    )
                elif isinstance(end_step, float):
                    current_step = self.current_epoch
                    value = (
                        (
                            start_value
                            + (end_value - start_value)
                            * max(
                                min(
                                    1.0,
                                    (current_step - start_step)
                                    / (end_step - start_step),
                                ),
                                0.0,
                            )
                        )
                        if current_step >= start_step
                        else 0.0
                    )
        return value

    def preprocess_data(self, batch, stage):
        pass

    def on_train_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.train_dataloader().dataset

    def on_validation_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.val_dataloader().dataset

    def on_test_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.test_dataloader().dataset

    def on_predict_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.preprocess_data(batch, 'train')
        update_module_step(self.model, self.current_epoch, self.global_step)
        if self.C(self.config.system.loss.lambda_curvature) > 0:
            self.model.with_curvature_loss = True
        else:
            self.model.with_curvature_loss = False
        if self.C(self.config.system.loss.lambda_albedo_smoothness) > 0:
            self.model.jitter_materials = True
        else:
            self.model.jitter_materials = False
        # Reinitialize per-frame occupancy grid at certain milestones
        if hasattr(self, "reinit_occupancy_grid_steps"):
            if self.global_step in self.reinit_occupancy_grid_steps:
                self.reinit_occupancy_grid()
                print("Re-create occupancy grid at step", self.global_step)

        # Reinitialize shape at certain milestones (for SMPL shape optimization)
        if hasattr(self, "reinit_shape_every_n_steps"):
            if self.reinit_shape_every_n_steps > 0 and self.global_step % self.reinit_shape_every_n_steps == 0:
                self.reinit_shape()

        # Reinitialize optimizer and scheduler at certain steps
        if hasattr(self, "reinit_optimizer_steps"):
            if self.global_step in self.reinit_optimizer_steps:
                self.trainer.strategy.setup_optimizers(self.trainer)
                print("Re-create optimizer and scheduler at step", self.global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.preprocess_data(batch, 'validation')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        # TODO: if we start `launch` with `mode=test`, then we need to call `update_module_step` first
        # in order to set correct status for e.g. coarse-to-fine HashGrid level, since `preprocess_data`
        # depends on correct initialization of `self.model` before it can preprocess any input data.
        # So we need to make sure:
        # 1) Calling this function during `mode=train` and calling this function with `mode=test` should
        #    have the same effect.
        # 2) We mannually set use epoch=250 and global_step=25000 for `mode=test`, but should figure out
        #    a way to directly load these values from checkpoint.
        update_module_step(self.model, 250, 25000)
        self.preprocess_data(batch, 'test')

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self.preprocess_data(batch, 'predict')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """

    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret
