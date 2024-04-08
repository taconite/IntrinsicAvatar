import torch
import models

from torch.autograd import grad

from systems.utils import update_module_step
from models.base import BaseModel
from models.network_utils import get_encoding, get_mlp


class BaseNonRigidDeformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def prepare_bbox(self, bbox):
        if hasattr(self, "bbox"):
            return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox


@models.register('dummy_non_rigid_deformer')
class DummyNonRigidDeformer(BaseNonRigidDeformer):
    def setup(self):
        pass

    def forward(self, points, cond, *args, with_jac=False):
        J_inv = torch.eye(points.shape[1], device=points.device).unsqueeze(0).repeat(
            points.shape[0], 1, 1
        )
        return points, J_inv


@models.register('non-rigid-deformer')
class NonRigidDeformer(BaseNonRigidDeformer):
    def setup(self):
        self.n_input_dims = 3
        self.n_output_dims = 3
        xyz_encoding_config = self.config.get('xyz_encoding_config', None)
        xyz_encoding = (
            get_encoding(self.n_input_dims, xyz_encoding_config)
            if xyz_encoding_config is not None
            else None
        )

        network = get_mlp(xyz_encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.xyz_encoding = xyz_encoding
        self.network = network

        self.kick_in_step = self.config.get('kick_in_step', 6000)
        self.enabled = False

    def forward(self, points, cond, *args, with_jac=False):
        if not self.enabled:
            J_inv = torch.eye(points.shape[1], device=points.device).unsqueeze(0).repeat(
                points.shape[0], 1, 1
            )
            return points, J_inv

        with torch.inference_mode(torch.is_inference_mode_enabled() and not with_jac):
            with torch.set_grad_enabled(self.training or with_jac):
                if with_jac:
                    if not self.training:
                        points = (
                            points.clone()
                        )  # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

            points_ = points # points in the original scale
            if self.xyz_encoding is not None:
                points = (points - self.center) / self.scale.clone() + 0.5
                xyz_embd = self.xyz_encoding(points.view(-1, 3))
            else:
                xyz_embd = torch.empty(
                    points.shape[:1] + (0,), dtype=points.dtype, device=points.device
                )

            network_inp = xyz_embd
            deformed_points = (
                points_
                + self.network(network_inp, cond)
                .view(*points.shape[:-1], self.n_output_dims)
                .float()
            )

            if with_jac:
                # Compute inverse jacobian
                jac = torch.zeros(
                    points.shape[0],
                    points.shape[1],
                    points.shape[1],
                    device=points.device,
                )
                for i in range(points.shape[1]):
                    jac[:, i, :] = grad(
                        deformed_points[:, i],
                        points_,
                        torch.ones_like(deformed_points[:, i]),
                        create_graph=True,
                        retain_graph=True,
                    )[0]

                J_inv = jac.detach().inverse()
            else:
                J_inv = torch.eye(points.shape[1], device=points.device).unsqueeze(0).repeat(
                    points.shape[0], 1, 1
                )

            return deformed_points, J_inv

    def update_step(self, epoch, global_step):
        update_module_step(self.xyz_encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)
        if global_step > self.kick_in_step:
            self.enabled = True
        else:
            self.enabled = False
