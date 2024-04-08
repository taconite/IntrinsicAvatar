import torch
import torch.nn as nn
import models

from models.base import BaseModel


@models.register('pose_correction')
class PoseCorrection(BaseModel):
    def setup(self):
        # Pose correction
        self.pose_correction = torch.nn.Embedding(self.config.dataset_length, 69)
        self.pose_correction.weight.data.zero_()
        self.shape_correction = torch.nn.Embedding(1, 10)
        self.shape_correction.weight.data.zero_()
        self.global_orient_correction = torch.nn.Embedding(
            self.config.dataset_length, 3
        )
        self.global_orient_correction.weight.data.zero_()
        self.transl_correction = torch.nn.Embedding(self.config.dataset_length, 3)
        self.transl_correction.weight.data.zero_()

        self.enable_pose_correction = False

    def forward(self, idx):
        # if self.training and self.enable_pose_correction:
        if self.enable_pose_correction:
            betas_correction = self.shape_correction(
                torch.zeros(1, device=idx.device).long()
            )
            global_orient_correction = self.global_orient_correction(idx)
            transl_correction = self.transl_correction(idx)
            pose_correction = self.pose_correction(idx)
        else:
            betas_correction = torch.zeros_like(self.shape_correction.weight)
            global_orient_correction = torch.zeros_like(
                self.global_orient_correction.weight[:1]
            )
            transl_correction = torch.zeros_like(self.transl_correction.weight[:1])
            pose_correction = torch.zeros_like(self.pose_correction.weight[:1])

        return {
            "betas_correction": betas_correction,
            "global_orient_correction": global_orient_correction,
            "transl_correction": transl_correction,
            "pose_correction": pose_correction,
        }

    def update_step(self, epoch, global_step):
        if (
            self.config.enable_pose_correction
            and global_step > self.config.pose_correction_start_step
        ):
            self.enable_pose_correction = True
