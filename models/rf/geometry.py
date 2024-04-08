import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import scale_anything, get_activation, cleanup, chunk_batch
from models.network_utils import get_encoding, get_mlp
from utils.misc import get_rank
from systems.utils import update_module_step


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')
        # self.radius = self.config.radius

    def prepare_bbox(self, bbox):
        if hasattr(self, "bbox"):
            return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax):
        def batch_func(x):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv

        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        bbox_cpu = self.bbox.cpu()
        bbox_np = bbox_cpu.numpy()
        mesh_coarse = self.isosurface_(bbox_np[0], bbox_np[1])
        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(bbox_cpu[0], bbox_cpu[1])
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(bbox_cpu[0], bbox_cpu[1])
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get("finite_difference_eps", 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == "finite_difference":
            print(
                f"Using finite difference to compute gradients with eps={self.finite_difference_eps}"
            )

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False, cond=torch.empty(0)):
        if points.numel() == 0:
            ret = [
                torch.empty(0, dtype=torch.float32, device=points.device)
            ]  # empty SDF
            if with_grad:
                ret.append(
                    torch.empty(0, 3, dtype=torch.float32, device=points.device)
                )  # empty gradient
            if with_feature:
                ret.append(
                    torch.empty(
                        0, self.n_output_dims, dtype=torch.float32, device=points.device
                    )
                )  # empty feature

            if with_laplace:
                ret.append(
                    torch.empty(0, dtype=torch.float32, device=points.device)
                )  # empty laplace

            return ret

        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = (points - self.center) / self.scale.clone() + 0.5

                out = self.network(self.encoding(points.view(-1, 3)), cond).view(*points.shape[:-1], self.n_output_dims).float()
                sdf, feature = out[...,0], out
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf,
                            points_,
                            grad_outputs=torch.ones_like(sdf),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        if with_laplace:
                            # curverture loss, from PermutoSDF
                            eps = 1e-4
                            rand_directions = torch.rand_like(points_)
                            rand_directions = F.normalize(rand_directions, dim=-1, eps=1e-6)
                            normal = F.normalize(grad, dim=-1, eps=1e-6)
                            tangent = torch.cross(normal, rand_directions)
                            rand_directions = tangent
                            points_d_ = (points_ + eps * rand_directions)
                            points_d = (points_d_ - self.center) / self.scale + 0.5

                            points_d_sdf = (
                                self.network(self.encoding(points_d.view(-1, 3)), cond)[..., 0]
                                .view(*points.shape[:-1])
                                .float()
                            )
                            grad_d = torch.autograd.grad(
                                points_d_sdf,
                                points_d_,
                                grad_outputs=torch.ones_like(points_d_sdf),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                            )[0]
                            dot = torch.sum(
                                F.normalize(grad, dim=-1, eps=1e-6)
                                * F.normalize(grad_d, dim=-1, eps=1e-6),
                                dim=-1,
                            )
                            angle = torch.acos(
                                torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
                            )  # goes to range 0 when the angle is the same and pi when is opposite
                            laplace = angle / np.pi
                    elif self.grad_type == 'finite_difference':
                        # TODO: finite difference is not compatible with the latest model
                        raise NotImplementedError("Finite difference is not implemented in the latest model")
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = (points_d_ - self.center) / self.scale + 0.5
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)), cond)[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps
                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points, cond=torch.empty(0)):
        points = (points - self.center) / self.scale + 0.5

        sdf = self.network(self.encoding(points.view(-1, 3)), cond).view(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)
        if self.grad_type == 'finite_difference':
            # TODO: finite difference is not compatible with the latest model
            raise NotImplementedError("Finite difference is not implemented in the latest model")
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    print(f"Update finite_difference_eps to {grid_size}")
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")
