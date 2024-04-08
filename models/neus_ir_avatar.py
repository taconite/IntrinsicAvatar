from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import models
from models.base import BaseModel
from models.utils import (
    chunk_batch,
    max_connected_component,
    check_off_in_surface_points_mesh,
    check_rays_off_in_surface_points_mesh
)
from models.deformers.snarf_deformer import get_bbox_from_smpl
from systems.utils import update_module_step
from lib.pbr import rgb_to_srgb
from nerfacc import (
    RayIntervals,
    OccGridEstimator,
    traverse_grids,
    render_visibility_from_alpha,
    render_visibility_from_density,
    render_weight_from_alpha,
    accumulate_along_rays,
)
from models.occ_grid.temporal_occ_grid import TemporalOccGridEstimator
from models.volrend import (
    rendering,
    rendering_with_normals_sdf,
    rendering_with_normals_mats_sdf,
)
from models.pbr.utils import sample_volume_interaction
from lib.nerfacc import (
    ContractionType,
    ray_resampling_merge,
    ray_resampling_fine,
    ray_resampling_sdf_fine,
    pack_info,
    pack_data,
    unpack_info,
    unpack_data,
)


@torch.no_grad()
def sampling_override(
    self,
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # sigma/alpha function for skipping invisible space
    sigma_fn: Optional[Callable] = None,
    alpha_fn: Optional[Callable] = None,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    t_min: Optional[Tensor] = None,  # [n_rays]
    t_max: Optional[Tensor] = None,  # [n_rays]
    # rendering options
    render_step_size: float = 1e-3,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sampling with spatial skipping.

    Note:
        This function is not differentiable to any inputs.

    Args:
        rays_o: Ray origins of shape (n_rays, 3).
        rays_d: Normalized ray directions of shape (n_rays, 3).
        sigma_fn: Optional. If provided, the marching will skip the invisible space
            by evaluating the density along the ray with `sigma_fn`. It should be a
            function that takes in samples {t_starts (N,), t_ends (N,),
            ray indices (N,)} and returns the post-activation density values (N,).
            You should only provide either `sigma_fn` or `alpha_fn`.
        alpha_fn: Optional. If provided, the marching will skip the invisible space
            by evaluating the density along the ray with `alpha_fn`. It should be a
            function that takes in samples {t_starts (N,), t_ends (N,),
            ray indices (N,)} and returns the post-activation opacity values (N,).
            You should only provide either `sigma_fn` or `alpha_fn`.
        near_plane: Optional. Near plane distance. Default: 0.0.
        far_plane: Optional. Far plane distance. Default: 1e10.
        t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
            If profided, the marching will start from maximum of t_min and near_plane.
        t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
            If profided, the marching will stop by minimum of t_max and far_plane.
        render_step_size: Step size for marching. Default: 1e-3.
        early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
        alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
        stratified: Whether to use stratified sampling. Default: False.
        cone_angle: Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.

    Returns:
        A tuple of {LongTensor, Tensor, Tensor}:

        - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
        - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
        - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

    Examples:

    .. code-block:: python

        >>> ray_indices, t_starts, t_ends = grid.sampling(
        >>>     rays_o, rays_d, render_step_size=1e-3)
        >>> t_mid = (t_starts + t_ends) / 2.0
        >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

    """

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    if t_min is not None:
        near_planes = torch.clamp(near_planes, min=t_min)
    if t_max is not None:
        far_planes = torch.clamp(far_planes, max=t_max)

    if stratified:
        near_planes += torch.rand_like(near_planes) * render_step_size

    intervals, samples, _ = traverse_grids(
        rays_o,
        rays_d,
        self.binaries,
        self.aabbs,
        near_planes=near_planes,
        far_planes=far_planes,
        step_size=render_step_size,
        cone_angle=cone_angle,
    )
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]
    ray_indices = samples.ray_indices
    packed_info = samples.packed_info

    # skip invisible space
    if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
        sigma_fn is not None or alpha_fn is not None
    ):
        alpha_thre = min(alpha_thre, self.occs.mean().item())

        # Compute visibility of the samples, and filter out invisible samples
        if sigma_fn is not None:
            if t_starts.shape[0] != 0:
                sigmas = sigma_fn(t_starts, t_ends, ray_indices)
            else:
                sigmas = torch.empty((0,), device=t_starts.device)
            assert (
                sigmas.shape == t_starts.shape
            ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
            masks = render_visibility_from_density(
                t_starts=t_starts,
                t_ends=t_ends,
                sigmas=sigmas,
                packed_info=packed_info,
                early_stop_eps=early_stop_eps,
                alpha_thre=alpha_thre,
            )
        elif alpha_fn is not None:
            if t_starts.shape[0] != 0:
                alphas = alpha_fn(t_starts, t_ends, ray_indices)
            else:
                alphas = torch.empty((0,), device=t_starts.device)
            assert (
                alphas.shape == t_starts.shape
            ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
            masks = render_visibility_from_alpha(
                alphas=alphas,
                packed_info=packed_info,
                early_stop_eps=early_stop_eps,
                alpha_thre=alpha_thre,
            )
        ray_indices, t_starts, t_ends = (
            ray_indices[masks],
            t_starts[masks],
            t_ends[masks],
        )
    return intervals, ray_indices, t_starts, t_ends


OccGridEstimator.sampling = sampling_override


def _meshgrid3d(
    res: Tensor, device: Union[torch.device, str] = "cpu"
) -> Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.long),
                torch.arange(res[1], dtype=torch.long),
                torch.arange(res[2], dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).to(device)


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.register_parameter(
            "variance", nn.Parameter(torch.tensor(self.config.init_val))
        )
        self.modulate = self.config.get("modulate", False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s

    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s

    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min(
                    (global_step / self.reach_max_steps)
                    * (self.max_inv_s - self.prev_inv_s)
                    + self.prev_inv_s,
                    self.max_inv_s,
                )


@models.register("neus-ir-avatar")
class NeuSIRAvatarModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.radiance = models.make(self.config.radiance.name, self.config.radiance)
        self.material = models.make(self.config.material.name, self.config.material)
        self.scatterer = models.make(self.config.scatterer.name, self.config.scatterer)
        self.pose_encoder = models.make(self.config.pose_encoder.name, self.config.pose_encoder)
        self.geometry.contraction_type = ContractionType.AABB
        self.emitter = models.make(self.config.light.name, self.config.light)

        self.material_feature = self.config.get("material_feature", "hybrid")
        assert self.material_feature in ["geometry", "radiance", "hybrid"], (
            "material_feature must be one of "
            + "['geometry', 'radiance', 'hybrid'], got %s" % self.material_feature
        )

        self.deformer = models.make(self.config.deformer.name, self.config.deformer)
        self.pose_correction = models.make(
            self.config.pose_correction.name, self.config.pose_correction
        )

        self.density = models.make(self.config.density.name, self.config.density)
        self.register_buffer(
            "scene_aabb",
            torch.as_tensor(
                self.config.scene_aabb,
                dtype=torch.float32,
            ),
        )
        scene_diag_len = torch.norm(
            self.scene_aabb[3:] - self.scene_aabb[:3]
        ).item()
        if self.config.grid_prune:
            self.occupancy_grid = TemporalOccGridEstimator(
                roi_aabb=self.scene_aabb[None, ...],
                resolution=64,
                levels=1,
            )
        self.randomized = self.config.randomized
        self.background_color = None
        self.samples_per_pixel = self.config.samples_per_pixel
        self.render_step_size = scene_diag_len / self.config.num_samples_per_ray
        self.num_samples_per_secondary_ray = self.config.get("num_samples_per_secondary_ray", 64)
        self.secondary_near_plane = self.config.get("secondary_near_plane", 0.05)
        self.secondary_far_plane = self.config.get("secondary_far_plane", 1.5)
        self.secondary_shader_chunk = self.config.get("secondary_shader_chunk", 160000)
        self.secondary_importance_sample = self.config.get("secondary_importance_sample", True)

        self.enable_phys = False if self.config.phys_kick_in_step > 0 else True
        self.opacity_eps = self.config.get("opacity_eps", 0.1)
        self.beta_iters = self.config.get("beta_iters", 10)
        self.zero_crossing_search = self.config.get("zero_crossing_search", True)
        self.volume_scattering = self.config.get("volume_scattering", True)

        self.albedo_only = False

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.radiance, epoch, global_step)
        update_module_step(self.density, epoch, global_step)
        update_module_step(self.pose_correction, epoch, global_step)

        if self.training and self.config.grid_prune:
            cond = self.pose_encoder(
                self.deformer.rot_mats, self.deformer.basic_joints
            )

            def occ_eval_fn(x):
                x, sdf, *_ = self.deformer(
                    x,
                    lambda x: self.geometry(
                        x,
                        with_grad=False,
                        with_feature=False,
                        with_laplace=False,
                        cond=cond,
                    ),
                    eval_mode=True,
                )
                density = self.density(sdf)
                alpha = 1.0 - torch.exp(-density * self.render_step_size)

                return alpha

            self.occupancy_grid.update_every_n_steps(
                step=global_step,
                t_idx=self.t_idx,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get("grid_prune_occ_thre", 0.01),
                ema_decay=self.config.get("grid_prune_ema_decay", 0.8),
                # warmup_steps=1e10,  # no warmup
                n=20,
            )

        # if not self.training or global_step >= self.config.phys_kick_in_step:
        if global_step >= self.config.phys_kick_in_step:
            self.enable_phys = True
        else:
            self.enable_phys = False

        if global_step > self.config.get("importance_sample_kick_in_step", 9000):
            self.importance_sample = True
        else:
            self.importance_sample = False

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def prepare(self, batch):
        batch.update(self.pose_correction(batch["index"]))
        self.deformer.prepare_deformer(batch)
        self.geometry.prepare_bbox(self.deformer.bbox)
        self.radiance.prepare_bbox(self.deformer.bbox)
        if not self.training:
            self.prepare_test_occupancy_grid()
            if self.enable_phys and "hdri" in batch:
                assert self.config.light.name == "envlight-tensor", (
                    "envlight-tensor is required for physically based rendering"
                    + " when hdri is provided"
                )
                self.emitter.base = nn.Parameter(batch["hdri"])
                self.emitter.pdf_scale = (
                    self.emitter.base.shape[0] * self.emitter.base.shape[1]
                ) / (2 * np.pi * np.pi)
                self.emitter.update_pdf()
                self.secondary_rays_d = self.deformer.transform_dirs_w2s(
                    self.emitter.sample(self.samples_per_pixel)
                )

    @torch.no_grad()
    def _compute_occupancy_grid(self, grid_coords, resolution=64):
        aabb = get_bbox_from_smpl(self.deformer.vertices).view(-1)

        # Randomly draw 3 samples in each voxel
        n_samples = 3
        grid_coords = grid_coords.unsqueeze(1).expand(-1, n_samples, -1)
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / resolution
        x = x.reshape(-1, 3)
        x = x * (aabb[3:] - aabb[:3]) + aabb[:3]
        cond = self.pose_encoder(self.deformer.rot_mats, self.deformer.basic_joints)

        def occ_eval_fn(x):
            x, sdf, *_ = self.deformer(
                x,
                lambda x: self.geometry(
                    x,
                    with_grad=False,
                    with_feature=False,
                    with_laplace=False,
                    cond=cond,
                ),
                eval_mode=True,
            )
            density = self.density(sdf)
            alpha = 1.0 - torch.exp(-density * self.render_step_size)

            return alpha

        occs = occ_eval_fn(x).reshape(-1, n_samples).max(1)[0]
        occs_ = F.max_pool3d(
            occs.reshape(1, 1, resolution, resolution, resolution),
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0].reshape(-1)
        thre = torch.clamp(
            occs_[occs_ >= 0].mean(), max=self.config.get("grid_prune_occ_thre", 0.01)
        )
        binaries = (occs_ > thre).reshape(1, resolution, resolution, resolution)

        # get maximum connected component
        mcc = max_connected_component(binaries)
        label = torch.mode(mcc[binaries.squeeze(0)], 0).values
        binaries = (mcc == label).reshape(binaries.shape)

        return occs, binaries, aabb

    def prepare_test_occupancy_grid(self):
        if hasattr(self, "occupancy_grid_test"):
            del self.occupancy_grid_test

        resolution = 64
        grid_coords = _meshgrid3d(
            torch.tensor(
                [resolution, resolution, resolution], device=self.rank
            ), device=self.rank
        ).reshape(resolution**3, 3)
        _, binaries, deformed_bbox = self._compute_occupancy_grid(
            grid_coords, resolution=resolution
        )

        occupancy_grid_test = OccGridEstimator(
            roi_aabb=deformed_bbox,
            resolution=resolution,
            levels=1,
        ).to(deformed_bbox.device)
        occupancy_grid_test.binaries = binaries

        self.occupancy_grid_test = occupancy_grid_test

    def compute_relative_smoothness_loss(self, values, values_jittor):

        base = torch.maximum(values, values_jittor).clip(min=1e-6)
        difference = torch.sum(((values - values_jittor) / base)**2, dim=-1, keepdim=True)  # [..., 1]

        return difference

    def get_alpha(self, sdf, normal, dirs, dists):
        density = self.density(sdf)
        alpha = 1.0 - torch.exp(-density * dists[:, 0])

        return alpha

    def compute_indirect_radiance(self, rays_o, rays_d):
        n_rays = rays_o.shape[0]

        def coarse_alpha_sdf_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * t_starts[..., None]
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)

            # Use chunk_batch to avoid OOM
            _, sdf, *others = chunk_batch(
                self.deformer,
                self.secondary_shader_chunk,
                False,
                positions,
                lambda x: self.geometry(
                    x,
                    with_grad=False,
                    with_feature=False,
                    with_laplace=False,
                    cond=self.cond,
                ),
                eval_mode=not self.training,
            )

            # sdf = torch.minimum(sdf[intervals.is_left], sdf[intervals.is_right])

            dists = (t_ends - t_starts)[..., None]
            # VolSDF does not need normal and t_dirs to compute alpha
            normal_world = None
            t_dirs = None

            alphas = self.get_alpha(sdf, normal_world, t_dirs, dists)
            return alphas, sdf

        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=t_origins.device
                ), torch.zeros((0,), device=t_origins.device)
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            positions, sdf, valid, sdf_grad, _, feature = self.deformer(
                positions,
                lambda x: self.geometry(
                    x,
                    with_grad=True,
                    with_feature=True,
                    with_laplace=False,
                    cond=self.cond,
                ),
                eval_mode=True,
            )
            dists = (t_ends - t_starts)[..., None]
            t_dirs = self.deformer.transform_dirs_s2w(t_dirs)
            normal_world = self.deformer.transform_dirs_s2w(sdf_grad)
            alphas = self.get_alpha(sdf, normal_world, t_dirs, dists)
            rgbs, *_ = self.radiance(positions, feature, t_dirs, normal_world)
            return sdf, rgbs, alphas

        secondary_render_step_size = (
            self.secondary_far_plane - self.secondary_near_plane
        ) / (self.num_samples_per_secondary_ray - 1)
        if self.training:
            intervals, ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
                rays_o,
                rays_d,
                # alpha_fn=alpha_fn,
                near_plane=self.secondary_near_plane,
                far_plane=self.secondary_far_plane,
                t_idx=self.t_idx,
                render_step_size=secondary_render_step_size,
                stratified=False,
            )
        else:
            intervals, ray_indices, t_starts, t_ends = self.occupancy_grid_test.sampling(
                rays_o,
                rays_d,
                # alpha_fn=alpha_fn,
                near_plane=self.secondary_near_plane,
                far_plane=self.secondary_far_plane,
                render_step_size=secondary_render_step_size,
                stratified=False,
            )

        # Importance sampling for secondary rays. Note that different to primary
        # rays, we only upsample once and use only importance samples while
        # discarding uniform samples
        if t_starts.numel() > 0 and self.secondary_importance_sample:
            for _ in range(1):
                with torch.no_grad():
                    alphas, sdfs = coarse_alpha_sdf_fn(t_starts, t_ends, ray_indices)
                    packed_info = pack_info(ray_indices=ray_indices, n_rays=n_rays)
                    if self.zero_crossing_search:
                        (
                            resampled_packed_info,
                            resampled_starts,
                            resampled_ends,
                            is_fg_sample,
                        ) = ray_resampling_sdf_fine(
                            packed_info,
                            t_starts[..., None],
                            t_ends[..., None],
                            alphas,
                            sdfs,
                            n_samples=4,
                        )
                    else:
                        weights, _ = render_weight_from_alpha(
                            alphas, ray_indices=ray_indices, n_rays=n_rays
                        )
                        (
                            resampled_packed_info,
                            resampled_starts,
                            resampled_ends,
                            is_fg_sample,
                        ) = ray_resampling_fine(
                            packed_info,
                            t_starts[..., None],
                            t_ends[..., None],
                            weights,
                            n_samples=4,
                        )

                # Keep only foreground samples
                resampled_ray_indices = unpack_info(
                    resampled_packed_info, len(resampled_starts)
                )
                ray_indices = resampled_ray_indices[is_fg_sample]
                t_starts = resampled_starts[is_fg_sample, 0]
                t_ends = resampled_ends[is_fg_sample, 0]

        (
            rgb_map,
            acc_map,
            _,
            extras,
        ) = rendering(
            t_starts,
            t_ends,
            ray_indices=ray_indices,
            n_rays=n_rays,
            rgb_alpha_fn=rgb_alpha_fn,
            render_bkgd=None,
            chunk_size=self.secondary_shader_chunk,
        )

        return 1.0 - acc_map, rgb_map

    def pbr_mis_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
           roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Scatterer sampling
            scatter_dirs = self.scatterer.sample(
                n=normal_interleaved,
                wi=wi_interleaved,
                alpha_x=roughness_interleaved.squeeze(-1),
                alpha_y=roughness_interleaved.squeeze(-1),
                albedo=albedo_interleaved,
                metallic=metallic_interleaved,
                attenuation=attenuation_interleaved,
            )
            # Light sampling
            if self.training:
                self.emitter.update_pdf()
            light_dirs = self.deformer.transform_dirs_w2s(
                self.emitter.sample(len(scatter_dirs))
            )
            # Concatenate all sampled directions
            secondary_rays_d = torch.cat([scatter_dirs, light_dirs], dim=0)
            secondary_rays_o = positions.repeat(2, 1)

            secondary_tr, secondary_rgb_map = self.compute_indirect_radiance(
                secondary_rays_o,
                secondary_rays_d,
            )

        # Compute pdf values of all rays (scatter + light)
        pdf_scatter = self.scatterer.pdf(
            n=normal_interleaved.repeat(2, 1),
            wi=wi_interleaved.repeat(2, 1),
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1).repeat(2),
            alpha_y=roughness_interleaved.squeeze(-1).repeat(2),
            albedo=albedo_interleaved.repeat(2, 1),
            metallic=metallic_interleaved.repeat(2, 1),
            attenuation=attenuation_interleaved.repeat(2, 1),
        )
        pdf_light = self.emitter.pdf(self.deformer.transform_dirs_s2w(secondary_rays_d))
        # Compute scatter ratio of all rays (scatter + light)
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff, spec = self.scatterer.eval(
            wi=wi_interleaved.repeat(2, 1),
            n=normal_interleaved.repeat(2, 1),
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1).repeat(2),
            alpha_y=roughness_interleaved.squeeze(-1).repeat(2),
            albedo=albedo_interleaved.repeat(2, 1),
            metallic=metallic_interleaved.repeat(2, 1),
            attenuation=attenuation_interleaved.repeat(2, 1),
        )
        # Query the environment map with all rays (scatter + light)
        em_li = self.emitter.eval(self.deformer.transform_dirs_s2w(secondary_rays_d))
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        # Compute MIS weights
        # We don't backpropagate through MIS weights, so we can use torch.where
        # without handling divide-by-zero cases
        mis_weights = torch.where(
            pdf_scatter + pdf_light > 1e-6,
            torch.reciprocal(pdf_scatter + pdf_light),
            torch.zeros_like(pdf_scatter),
        )
        # Combine scatterer and light samples using MIS weights
        # Note the PDF terms of MC integration in the denominator is
        # cancelled out by MIS weights
        Lo_diff = (Li * diff) * mis_weights
        Lo_spec = (Li * spec) * mis_weights

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd.repeat(2, 1) * Lo_diff + ks.repeat(2, 1) * Lo_spec

        # Sum over sampling strategies
        Lo = Lo.reshape(2, -1, 3).sum(dim=0)
        Lo_diff = Lo_diff.reshape(2, -1, 3).sum(dim=0)
        Lo_spec = Lo_spec.reshape(2, -1, 3).sum(dim=0)

        return Lo, Lo_diff, Lo_spec

    def pbr_uniform_light_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
        n_rays,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
           roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Stratified light sampling
            # Since we sample uniformly on the light sphere, we do not need to
            # convert the sampled directions to the SMPL space
            (
                secondary_rays_d,
                inv_pdf,
            ) = self.emitter.sample_uniform_sphere_stratified(
                n_rays,
                16,
                32,
                device=albedo_interleaved.device,
            )

            secondary_rays_o = positions.reshape(-1, 3)
            cosine_mask = (normal_interleaved * secondary_rays_d).sum(dim=-1) > 1e-6

            secondary_tr = torch.zeros(len(cosine_mask), 1, device=cosine_mask.device)
            secondary_rgb_map = torch.zeros(
                len(cosine_mask), 3, device=cosine_mask.device
            )

            if cosine_mask.sum() > 0:
                (
                    secondary_tr[cosine_mask],
                    secondary_rgb_map[cosine_mask],
                ) = self.compute_indirect_radiance(
                    secondary_rays_o[cosine_mask], secondary_rays_d[cosine_mask]
                )
                secondary_tr.clamp_(0.0, 1.0)

            tr_mask = secondary_tr[..., 0] > 0.0

        # Compute scatter ratio of all rays
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff = torch.zeros_like(albedo_interleaved[..., :1])
        spec = torch.zeros_like(albedo_interleaved)
        if cosine_mask.sum() > 0:
            diff[cosine_mask], spec[cosine_mask] = self.scatterer.eval(
                wi=wi_interleaved[cosine_mask],
                n=normal_interleaved[cosine_mask],
                wo=secondary_rays_d[cosine_mask],
                alpha_x=roughness_interleaved.squeeze(-1)[cosine_mask],
                alpha_y=roughness_interleaved.squeeze(-1)[cosine_mask],
                albedo=albedo_interleaved[cosine_mask],
                metallic=metallic_interleaved[cosine_mask],
                attenuation=attenuation_interleaved[cosine_mask],
            )
        # Query the environment map with all rays
        em_li = torch.zeros_like(secondary_rgb_map)
        if cosine_mask.sum() > 0:
            em_li[cosine_mask & tr_mask] = self.emitter.eval(
                self.deformer.transform_dirs_s2w(
                    secondary_rays_d[cosine_mask & tr_mask]
                )
            )
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        # Compute diffuse/specular component of the outgoing radiance
        Lo_diff = Li * diff * inv_pdf
        Lo_spec = Li * spec * inv_pdf
        vis = 2 * torch.ones_like(em_li) * secondary_tr

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd * Lo_diff + ks * Lo_spec

        return Lo, Lo_diff, Lo_spec, vis

    def pbr_light_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
        shuffled_indices,
        n_rays,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
           roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Stratified light sampling
            if self.training:
                self.emitter.update_pdf()

            if self.training:
                secondary_rays_d = self.deformer.transform_dirs_w2s(
                    self.emitter.sample(len(albedo_interleaved))
                )
            else:
                secondary_rays_d = self.secondary_rays_d.repeat(n_rays, 1)[
                    shuffled_indices
                ]

            secondary_rays_o = positions.reshape(-1, 3)
            cosine_mask = (normal_interleaved * secondary_rays_d).sum(dim=-1) > 1e-6

            secondary_tr = torch.zeros(len(cosine_mask), 1, device=cosine_mask.device)
            secondary_rgb_map = torch.zeros(
                len(cosine_mask), 3, device=cosine_mask.device
            )

            if cosine_mask.sum() > 0:
                (
                    secondary_tr[cosine_mask],
                    secondary_rgb_map[cosine_mask],
                ) = self.compute_indirect_radiance(
                    secondary_rays_o[cosine_mask], secondary_rays_d[cosine_mask]
                )
                secondary_tr.clamp_(0.0, 1.0)

            tr_mask = secondary_tr[..., 0] > 0.0

        # Compute scatter ratio of all rays
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff = torch.zeros_like(albedo_interleaved[..., :1])
        spec = torch.zeros_like(albedo_interleaved)
        if cosine_mask.sum() > 0:
            diff[cosine_mask], spec[cosine_mask] = self.scatterer.eval(
                wi=wi_interleaved[cosine_mask],
                n=normal_interleaved[cosine_mask],
                wo=secondary_rays_d[cosine_mask],
                alpha_x=roughness_interleaved.squeeze(-1)[cosine_mask],
                alpha_y=roughness_interleaved.squeeze(-1)[cosine_mask],
                albedo=albedo_interleaved[cosine_mask],
                metallic=metallic_interleaved[cosine_mask],
                attenuation=attenuation_interleaved[cosine_mask],
            )
        # Query the environment map with all rays
        em_li = torch.zeros_like(secondary_rgb_map)
        if cosine_mask.sum() > 0:
            em_li[cosine_mask & tr_mask] = self.emitter.eval(
                self.deformer.transform_dirs_s2w(
                    secondary_rays_d[cosine_mask & tr_mask]
                )
            )
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr

        pdf = torch.ones_like(em_li[..., :1])
        pdf[cosine_mask & tr_mask] = self.emitter.pdf(
            self.deformer.transform_dirs_s2w(secondary_rays_d[cosine_mask & tr_mask])
        )
        # Avoid divide-by-zero cases
        pdf = torch.where(pdf > 0, pdf, torch.ones_like(pdf))
        # Compute diffuse/specular component of the outgoing radiance
        Lo_diff = Li * diff / pdf
        Lo_spec = Li * spec / pdf

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd * Lo_diff + ks * Lo_spec

        return Lo, Lo_diff, Lo_spec

    def pbr_mats_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
           roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Scatterer sampling
            secondary_rays_d = self.scatterer.sample(
                n=normal_interleaved,
                wi=wi_interleaved,
                alpha_x=roughness_interleaved.squeeze(-1),
                alpha_y=roughness_interleaved.squeeze(-1),
                albedo=albedo_interleaved,
                metallic=metallic_interleaved,
                attenuation=attenuation_interleaved,
            )
            secondary_rays_o = positions.reshape(-1, 3)

            secondary_tr, secondary_rgb_map = self.compute_indirect_radiance(
                secondary_rays_o,
                secondary_rays_d,
            )

        # Compute pdf values of all rays
        pdf = self.scatterer.pdf(
            n=normal_interleaved,
            wi=wi_interleaved,
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1),
            alpha_y=roughness_interleaved.squeeze(-1),
            albedo=albedo_interleaved,
            metallic=metallic_interleaved,
            attenuation=attenuation_interleaved,
        )
        # Avoid divide-by-zero cases
        pdf = torch.where(pdf > 0, pdf, torch.ones_like(pdf))
        # Compute scatter ratio of all rays
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff, spec = self.scatterer.eval(
            wi=wi_interleaved,
            n=normal_interleaved,
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1),
            alpha_y=roughness_interleaved.squeeze(-1),
            albedo=albedo_interleaved,
            metallic=metallic_interleaved,
            attenuation=attenuation_interleaved,
        )
        # Query the environment map with all rays (scatter + light)
        em_li = self.emitter.eval(self.deformer.transform_dirs_s2w(secondary_rays_d))
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        # Combine scatterer and light samples using MIS weights
        # Note the PDF terms of MC integration in the denominator is
        # cancelled out by MIS weights
        Lo_diff = Li * diff / pdf
        Lo_spec = Li * spec / pdf

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd * Lo_diff + ks * Lo_spec

        return Lo, Lo_diff, Lo_spec

    def forward_(self, rays):
        rays = self.deformer.transform_rays_w2s(rays)
        n_rays = rays.shape[0]
        rays_o, rays_d, near, far = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]

        def coarse_alpha_fn(intervals, dists=None, sdf_prev=None, is_resampled=None):
            if is_resampled is None:
                t_origins = rays_o[intervals.ray_indices]
                t_dirs = rays_d[intervals.ray_indices]
                positions = t_origins + t_dirs * intervals.vals[..., None]
            else:
                t_origins = rays_o[intervals.ray_indices[is_resampled]]
                t_dirs = rays_d[intervals.ray_indices[is_resampled]]
                positions = t_origins + t_dirs * intervals.vals[is_resampled][..., None]
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device), torch.zeros(
                    (0,), device=t_origins.device
                )
            _, sdf_curr, *others = self.deformer(
                positions,
                lambda x: self.geometry(
                    x,
                    with_grad=False,
                    with_feature=False,
                    with_laplace=False,
                    cond=self.cond,
                ),
                eval_mode=not self.training,
            )

            if sdf_prev is not None:
                sdf = torch.ones_like(intervals.vals) * 1e10
                sdf[is_resampled] = sdf_curr
                sdf[~is_resampled] = sdf_prev
            else:
                sdf = sdf_curr

            sdf_merge = torch.ones_like(intervals.vals) * 1e10
            sdf_min = torch.minimum(sdf[intervals.is_left], sdf[intervals.is_right])
            sdf_merge[intervals.is_left] = sdf_min

            if dists is None:
                dists = (self.render_step_size * torch.ones_like(intervals.vals))

            # VolSDF does not need normal and t_dirs to compute alpha
            normal_world = None
            t_dirs = None

            alphas = self.get_alpha(sdf_merge, normal_world, t_dirs, dists[..., None])
            return alphas, sdf_merge

        def alpha_fn(intervals):
            ray_indices = intervals.ray_indices[intervals.is_left]
            t_starts = intervals.vals[intervals.is_left]
            t_ends = intervals.vals[intervals.is_right]
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)
            _, sdf_curr, *others = self.deformer(
                positions,
                lambda x: self.geometry(
                    x,
                    with_grad=False,
                    with_feature=False,
                    with_laplace=False,
                    cond=self.cond,
                ),
                eval_mode=not self.training,
            )

            sdf = torch.ones_like(intervals.vals) * 1e10
            sdf[intervals.is_left] = sdf_curr
            dists = torch.zeros_like(intervals.vals)
            dists[intervals.is_left] = t_ends - t_starts

            # VolSDF does not need normal and t_dirs to compute alpha
            normal_world = None
            t_dirs = None

            alphas = self.get_alpha(sdf, normal_world, t_dirs, dists[..., None])
            return alphas

        def rgb_normal_alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=t_origins.device
                ), torch.zeros((0,), device=t_origins.device)
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            ret = self.deformer(
                positions,
                lambda x: self.geometry(
                    x,
                    with_grad=True,
                    with_feature=True,
                    with_laplace=self.training and self.with_curvature_loss,
                    cond=self.cond,
                ),
                eval_mode=not self.training,
            )
            positions, sdf, valid, sdf_grad, sdf_grad_cano, feature = ret[:6]
            laplace = ret[6] if len(ret) > 6 else torch.zeros_like(sdf)
            dists = (t_ends - t_starts)[..., None]
            normal_smpl = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            normal_world = self.deformer.transform_dirs_s2w(sdf_grad)
            t_dirs = self.deformer.transform_dirs_s2w(t_dirs)
            alphas = self.get_alpha(sdf, normal_world, t_dirs, dists)
            rgbs, *_ = self.radiance(positions, feature, t_dirs, normal_world)
            return positions, valid, rgbs, normal_smpl, normal_world, alphas, sdf, sdf_grad, laplace

        def rgb_normal_mats_alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros((0, 3), device=t_origins.device), torch.zeros(
                    (0,), device=t_origins.device
                )
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0

            # Get geometry related outputs
            ret = self.deformer(
                positions,
                lambda x: self.geometry(
                    x,
                    with_grad=True,
                    with_feature=True,
                    with_laplace=self.training and self.with_curvature_loss,
                    cond=self.cond,
                ),
                eval_mode=not self.training,
            )
            positions, sdf, valid, sdf_grad, sdf_grad_cano, feature = ret[:6]
            laplace = ret[6] if len(ret) > 6 else torch.zeros_like(sdf)
            dists = (t_ends - t_starts)[..., None]
            normal_smpl = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            normal_world = self.deformer.transform_dirs_s2w(sdf_grad)
            t_dirs = self.deformer.transform_dirs_s2w(t_dirs)
            alphas = self.get_alpha(sdf, normal_world, t_dirs, dists)

            # Get radiance
            rgbs, rgb_feature = self.radiance(positions, feature, t_dirs, normal_world)

            if self.material_feature == "geometry":
                # `feature` stays the same
                pass
            elif self.material_feature == "radiance":
                # `feature` is replaced by `rgb_feature`
                feature = rgb_feature
            elif self.material_feature == "hybrid":
                # `feature` is the concatenation of `rgb_feature` and `feature`
                feature = torch.cat([rgb_feature, feature], dim=-1)

            # Get materials
            materials = self.material(feature)
            if not self.training and hasattr(self, "albedo_align_ratio"):
                materials[..., :3] = materials[..., :3] * self.albedo_align_ratio
            if self.training and self.jitter_materials:
                positions_jitter = positions + torch.randn_like(positions) * 0.01
                _, feature_jitter = self.geometry(
                    positions_jitter, with_grad=False, with_feature=True, cond=self.cond
                )
                if self.material_feature == "geometry":
                    pass
                elif self.material_feature == "radiance":
                    feature_jitter = self.radiance(
                        positions_jitter, feature_jitter, t_dirs, feature_only=True
                    )
                elif self.material_feature == "hybrid":
                    feature_jitter = torch.cat(
                        [
                            self.radiance(
                                positions_jitter,
                                feature_jitter,
                                t_dirs,
                                feature_only=True,
                            ),
                            feature_jitter,
                        ],
                        dim=-1,
                    )
                materials_jitter = self.material(feature_jitter)
            else:
                materials_jitter = torch.zeros_like(materials)

            return (
                positions,
                valid,
                rgbs,
                normal_smpl,
                normal_world,
                materials,
                materials_jitter,
                alphas,
                sdf,
                sdf_grad,
                laplace,
            )

        if self.training:
            intervals, ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
                rays_o,
                rays_d,
                # alpha_fn=alpha_fn,
                # t_min=near,
                # t_max=far,
                render_step_size=self.render_step_size,
                t_idx=self.t_idx,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0,
            )
        else:
            intervals, ray_indices, t_starts, t_ends = self.occupancy_grid_test.sampling(
                rays_o,
                rays_d,
                # alpha_fn=alpha_fn,
                # t_min=near,
                # t_max=far,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0,
            )

        # Importance sampling
        if self.importance_sample and t_starts.numel() > 0:
            resampled_dists = None
            sdf = None
            is_resampled = None
            for upsample_iter in range(2):
                with torch.no_grad():
                    if upsample_iter == 0:
                        alphas, sdf = coarse_alpha_fn(
                            intervals, resampled_dists, sdf, is_resampled
                        )
                    else:
                        alphas = alpha_fn(intervals)
                    # if alphas.numel() == 0 or sdf.numel() == 0:
                    #     break
                    weights, _ = render_weight_from_alpha(
                        alphas, ray_indices=intervals.ray_indices, n_rays=n_rays
                    )
                    # packed_info = pack_info(ray_indices=ray_indices, n_rays=n_rays)
                    (
                        resampled_packed_info,
                        resampled_vals,
                        resampled_dists,
                        resampled_is_left,
                        resampled_is_right,
                        is_resampled,
                        is_fg_sample,
                    ) = ray_resampling_merge(
                        intervals.packed_info.int(),
                        intervals.vals,
                        intervals.is_left,
                        intervals.is_right,
                        weights,
                        n_samples=16,
                    )

                # Keep only foreground samples
                resampled_ray_indices = unpack_info(
                    resampled_packed_info, len(resampled_vals)
                )[is_fg_sample]
                resampled_packed_info = pack_info(resampled_ray_indices, n_rays)
                resampled_dists = resampled_dists[is_fg_sample]
                is_resampled = is_resampled[is_fg_sample]

                # t_starts = resampled_vals[resampled_is_left]
                # t_ends = resampled_vals[resampled_is_right]
                # ray_indices = resampled_ray_indices[resampled_is_left[is_fg_sample]]

                intervals = RayIntervals(
                        vals=resampled_vals[is_fg_sample],
                        is_left=resampled_is_left[is_fg_sample],
                        is_right=resampled_is_right[is_fg_sample],
                        ray_indices=resampled_ray_indices,
                        packed_info=resampled_packed_info,
                    )

                # new_intervals = RayIntervals(
                #         vals=resampled_vals[is_fg_sample],
                #         is_left=resampled_is_left[is_fg_sample],
                #         is_right=resampled_is_right[is_fg_sample],
                #         ray_indices=resampled_ray_indices,
                #         packed_info=resampled_packed_info,
                #     )

                # # Validate outputs
                # for s, e in new_intervals.packed_info:
                #     e = s + e
                #     ray_vals = new_intervals.vals[s:e]
                #     assert (ray_vals[1:] >= ray_vals[:-1]).all()
                #     assert (ray_vals[1:] - ray_vals[:-1] == resampled_dists[s:e-1]).all()

                # unique_indices = torch.unique(resampled_ray_indices)
                # for ray_index in unique_indices:
                #     mask_tgt = intervals.ray_indices == ray_index
                #     mask = new_intervals.ray_indices == ray_index
                #     vals_tgt = intervals.vals[mask_tgt]
                #     vals_ = new_intervals.vals[mask]
                #     assert (
                #         vals_tgt[intervals.is_left[mask_tgt]]
                #         == vals_[(~is_resampled)[mask]][
                #             new_intervals.is_left[mask][(~is_resampled)[mask]]
                #         ].reshape(-1)
                #     ).all()
                #     assert (
                #         vals_tgt[intervals.is_right[mask_tgt]]
                #         == vals_[(~is_resampled)[mask]][
                #             new_intervals.is_right[mask][(~is_resampled)[mask]]
                #         ].reshape(-1)
                #     ).all()

                # intervals = new_intervals

        # if not self.training and self.importance_sample and intervals.vals.numel() > 100000:
        #     print ("Warning: too many samples, skip importance sampling")
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = intervals.ray_indices[intervals.is_left]

        midpoints = (t_starts + t_ends) / 2.
        intervals = t_ends - t_starts

        if self.enable_phys:
            (
                rgb_map,
                normal_map,
                albedo_map,
                roughness_map,
                metallic_map,
                acc_map,
                depth_map,
                extras,
            ) = rendering_with_normals_mats_sdf(
                t_starts,
                t_ends,
                ray_indices=ray_indices,
                n_rays=n_rays,
                rgb_alpha_fn=rgb_normal_mats_alpha_fn,
                render_bkgd=None,
                material_dim=self.material.n_output_dims
            )
            rgb_phys_map = torch.zeros_like(rgb_map)
            demod_phys_map = torch.zeros_like(rgb_map)
            if self.config.render_mode == "uniform_light":
                visibility_map = torch.zeros(n_rays, 1, device=rays_o.device)
        else:
            (
                rgb_map,
                normal_map,
                acc_map,
                depth_map,
                extras,
            ) = rendering_with_normals_sdf(
                t_starts,
                t_ends,
                ray_indices=ray_indices,
                n_rays=n_rays,
                rgb_alpha_fn=rgb_normal_alpha_fn,
                render_bkgd=None,
            )

        depth_map = depth_map + (1. - acc_map) * far[..., None]

        # Material-based surface scattering
        valid_indices = torch.nonzero(acc_map > 0)[..., 0]
        if valid_indices.numel() > 0 and self.enable_phys and not self.albedo_only:
            weights = extras["weights"]
            normals_smpl = extras["normals"]
            normal_map_smpl = accumulate_along_rays(
                weights, values=normals_smpl, ray_indices=ray_indices, n_rays=n_rays
            )
            n_valid = len(valid_indices)
            valid_normal = F.normalize(normal_map_smpl[valid_indices], dim=-1)
            normal_interleaved = valid_normal.repeat_interleave(
                self.samples_per_pixel, dim=0
            ).reshape(-1, 3)
            valid_albedo = albedo_map[valid_indices]
            albedo_interleaved = valid_albedo.repeat_interleave(
                self.samples_per_pixel, dim=0
            ).reshape(-1, 3)
            valid_roughness = roughness_map[valid_indices]
            roughness_interleaved = valid_roughness.repeat_interleave(
                self.samples_per_pixel, dim=0
            ).reshape(-1, 1)
            valid_metallic = metallic_map[valid_indices]
            metallic_interleaved = valid_metallic.repeat_interleave(
                self.samples_per_pixel, dim=0
            ).reshape(-1, 1)
            valid_dirs = rays_d[valid_indices]
            wi_interleaved = valid_dirs.repeat_interleave(
                self.samples_per_pixel, dim=0
            ).reshape(-1, 3)
            valid_origins = rays_o[valid_indices]
            valid_depth = depth_map[valid_indices]
            positions_interleaved = (
                (valid_origins + valid_dirs * valid_depth)
                .repeat_interleave(self.samples_per_pixel, dim=0)
                .reshape(-1, 3)
            )

            # if self.config.render_mode in ["mats", "mis"]:
            #     fg_Lo, fg_Lo_diff, fg_Lo_spec = eval(
            #         "self.pbr_" + self.config.render_mode + "_forward"
            #     )(
            #         resampled_extras["normals"],
            #         resampled_extras["albedo"],
            #         resampled_extras["roughness"],
            #         resampled_extras["metallic"],
            #         resampled_extras["positions"],
            #         resampled_extras["t_dirs"],
            #     )
            if self.config.render_mode == "uniform_light":
                assert self.samples_per_pixel == 512
                Lo, Lo_diff, Lo_spec, vis = eval(
                    "self.pbr_" + self.config.render_mode + "_forward"
                )(
                    normal_interleaved,
                    albedo_interleaved,
                    roughness_interleaved,
                    metallic_interleaved,
                    positions_interleaved,
                    wi_interleaved,
                    n_rays=n_valid,
                )
                Lo = Lo.reshape(n_valid, self.samples_per_pixel, -1).mean(dim=1)
                Lo_diff = Lo_diff.reshape(n_valid, self.samples_per_pixel, -1).mean(dim=1)
                Lo_spec = Lo_spec.reshape(n_valid, self.samples_per_pixel, -1).mean(dim=1)
                vis = (
                    vis.reshape(n_valid, self.samples_per_pixel, -1)
                    .mean(dim=1)
                    .mean(dim=-1, keepdim=True)
                )
                visibility_map.scatter_(0, valid_indices[..., None].expand_as(vis), vis)
            else:
                raise NotImplementedError(
                    f"Render mode {self.config.render_mode} not supported."
                )

            rgb_phys_map.scatter_(
                0,
                valid_indices[..., None].expand_as(Lo),
                Lo,
            )
            demod_phys_map = demod_phys_map.scatter_(
                0,
                valid_indices[..., None].expand_as(Lo),
                Lo_diff + Lo_spec,
            )
        elif self.enable_phys:
            rgb_phys_map = torch.zeros(
                (n_rays, 3), device=rays_o.device
            )
            demod_phys_map = rgb_phys_map

        out = {
            "comp_rgb": rgb_map,
            # "comp_normal": torch.einsum("ij,bj->bi", self.opencv2opengl, normal_map),
            "comp_normal": normal_map,
            "opacity": acc_map,
            "depth": depth_map,
            "rays_valid": acc_map > 0,
            "rays_valid_phys": acc_map > 0
            if self.enable_phys
            else torch.zeros_like(acc_map, dtype=torch.bool),
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rays.device
            ),
        }

        if self.enable_phys:
            out.update(
                {
                    "comp_rgb_phys": rgb_phys_map,
                    "comp_demod_phys": demod_phys_map,
                    "comp_albedo": albedo_map,
                    "comp_metallic": metallic_map,
                    "comp_roughness": roughness_map,
                }
            )
            if self.config.render_mode == "uniform_light":
                out.update(
                    {
                        "visibility": visibility_map
                        + (1.0 - acc_map) * torch.zeros_like(visibility_map)
                    }
                )

        # if not self.training:
        #     smpl_mask = torch.zeros_like(rgb_map[..., :1])
        #     positions_cano = extras["positions"]
        #     valid = extras["valid"]
        #     if positions_cano.numel() and valid.sum() > 0:
        #         (
        #             index_off_surface,
        #             index_in_surface,
        #         ) = check_rays_off_in_surface_points_mesh(
        #             self.deformer.vs_template,
        #             self.deformer.body_model.faces_tensor,
        #             positions_cano[valid],
        #             ray_indices[valid],
        #             n_rays
        #         )
        #         smpl_mask[index_in_surface] = 1.0
        #     out.update({"smpl_mask": smpl_mask})

        if self.training:
            weights = extras["weights"]
            sdf = extras["sdf"]
            sdf_grad = extras["sdf_grad"]
            sdf_laplace = extras["laplace"]
            out.update(
                {
                    "sdf_samples": sdf,
                    "sdf_grad_samples": sdf_grad,
                    "sdf_laplace_samples": sdf_laplace,
                    "weights": weights.view(-1),
                    # TODO: following variables are useful for unbounded scenes
                    "points": midpoints.view(-1),
                    "intervals": intervals.view(-1),
                    "ray_indices": ray_indices.view(-1),
                }
            )
            if self.with_in_surface_loss:
                positions_cano = extras["positions"]
                valid = extras["valid"]
                (
                    index_off_surface,
                    index_in_surface,
                ) = check_off_in_surface_points_mesh(
                    self.deformer.vs_template,
                    self.deformer.body_model.faces_tensor,
                    self.deformer.faces_to_include,
                    positions_cano[valid],
                )
                out.update({
                    "index_off_surface": index_off_surface.view(-1),
                    "index_in_surface": index_in_surface.view(-1),
                    "valid": valid.view(-1),
                })

            if ray_indices.numel() > 0 and self.enable_phys:
                normals = extras["normals"]
                albedo = extras["albedo"]
                roughness = extras["roughness"]
                metallic = extras["metallic"]
                albedo_jitter = extras["albedo_jitter"]
                roughness_jitter = extras["roughness_jitter"]
                metallic_jitter = extras["metallic_jitter"]

                # Normal orientation loss and material smoothness losses
                normals_orientation_loss = torch.sum(
                    rays_d[ray_indices] * normals, dim=-1, keepdim=True
                ).clamp(min=0)
                albedo_smoothness_loss = self.compute_relative_smoothness_loss(
                    albedo, albedo_jitter
                )
                roughness_smoothness_loss = self.compute_relative_smoothness_loss(
                    roughness, roughness_jitter
                )
                metallic_smoothness_loss = self.compute_relative_smoothness_loss(
                    metallic, metallic_jitter
                )

                normals_orientation_loss_map = accumulate_along_rays(
                    weights,
                    values=normals_orientation_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                albedo_smoothness_loss_map = accumulate_along_rays(
                    weights,
                    values=albedo_smoothness_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                roughness_smoothness_loss_map = accumulate_along_rays(
                    weights,
                    values=roughness_smoothness_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                metallic_smoothness_loss_map = accumulate_along_rays(
                    weights,
                    values=metallic_smoothness_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
            else:
                normals_orientation_loss_map = torch.zeros_like(rgb_map[..., :1])
                albedo_smoothness_loss_map = torch.zeros_like(rgb_map[..., :1])
                roughness_smoothness_loss_map = torch.zeros_like(rgb_map[..., :1])
                metallic_smoothness_loss_map = torch.zeros_like(rgb_map[..., :1])

            out.update(
                {
                    "normals_orientation_loss_map": normals_orientation_loss_map,
                    "albedo_smoothness_loss_map": albedo_smoothness_loss_map,
                    "roughness_smoothness_loss_map": roughness_smoothness_loss_map,
                    "metallic_smoothness_loss_map": metallic_smoothness_loss_map,
                }
            )

        if self.config.learned_background:
            raise NotImplementedError("Learned background not implemented.")
        else:
            out_bg = {
                "comp_rgb": self.background_color[None, :].expand(*rgb_map.shape),
                "num_samples": torch.zeros_like(out["num_samples"]),
                "rays_valid": torch.zeros_like(out["rays_valid"]),
                "rays_valid_phys": torch.zeros_like(
                    out["rays_valid_phys"], dtype=torch.bool
                ),
            }
            if self.enable_phys:
                out_bg.update(
                    {
                        "comp_albedo": torch.zeros(
                            (1, 3), dtype=torch.float32, device=rays_o.device
                        ).expand(*albedo_map.shape),
                        "comp_metallic": self.background_color[None, :]
                        .mean(-1, keepdim=True)
                        .expand(metallic_map.shape),
                        "comp_roughness": self.background_color[None, :]
                        .mean(-1, keepdim=True)
                        .expand(roughness_map.shape),
                    }
                )

        out_full = {
            "comp_rgb": rgb_to_srgb(
                out["comp_rgb"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])
            ).clamp(0, 1),
            "num_samples": out["num_samples"] + out_bg["num_samples"],
            "rays_valid": out["rays_valid"] | out_bg["rays_valid"],
            "rays_valid_phys": out["rays_valid_phys"] | out_bg["rays_valid_phys"],
        }
        if self.enable_phys:
            out_full.update(
                {
                    "comp_rgb_phys": rgb_to_srgb(out["comp_rgb_phys"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])).clamp(0, 1),
                    "comp_demod_phys": rgb_to_srgb(out["comp_demod_phys"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])).clamp(0, 1),
                    "comp_albedo": out["comp_albedo"]
                    + out_bg["comp_albedo"] * (1.0 - out["opacity"]),
                    "comp_metallic": out["comp_metallic"]
                    + out_bg["comp_metallic"] * (1.0 - out["opacity"]),
                    "comp_roughness": out["comp_roughness"]
                    + out_bg["comp_roughness"] * (1.0 - out["opacity"]),
                }
            )

        return {
            **out,
            **{k + "_bg": v for k, v in out_bg.items()},
            **{k + "_full": v for k, v in out_full.items()},
        }

    def forward(self, rays):
        self.cond = self.pose_encoder(
            self.deformer.rot_mats, self.deformer.basic_joints
        )
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(
                self.forward_,
                self.config.ray_chunk,
                True,
                rays,
            )
        return {**out, "beta": self.density.get_beta()}

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.radiance.regularizations(out))
        if self.enable_phys:
            losses.update(self.material.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(
                self.geometry,
                export_config.chunk_size,
                False,
                mesh["v_pos"].to(self.rank),
                with_grad=True,
                with_feature=True,
            )
            # normal = F.normalize(sdf_grad, p=2, dim=-1)
            # rgb = self.radiance(
            #     feature, -normal, normal
            # )  # set the viewing directions to the normal to get "albedo"
            # mesh["v_rgb"] = rgb.cpu()
        return mesh
