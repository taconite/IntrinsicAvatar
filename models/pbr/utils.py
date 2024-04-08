from typing import Callable

import torch

from torch import Tensor

from nerfacc import render_weight_from_alpha, accumulate_along_rays

from lib.nerfacc import (
    pack_info,
    unpack_info,
    ray_resampling,
    ray_resampling_sdf_fine,
)


# def compute_indirect_radiance(
#     rays_o: Tensor,
#     rays_d: Tensor,
#     near_plane: float,
#     far_plane: float,
#     num_samples_per_ray: int,
#     grid_estimator: Union[OccGridEstimator, PropNetEstimator],
#     shader_chunk: Optional[int] = 160000,
#     rgb_sigma_fn: Optional[Callable] = None,
#     rgb_alpha_fn: Optional[Callable] = None,
#     sigma_fn: Optional[Callable] = None,
#     alpha_fn: Optional[Callable] = None,
# ):
#     n_rays = rays_o.shape[0]
#     # The user must provide both color and geometry functions based on either sigma or alpha
#     assert (rgb_sigma_fn is not None and sigma_fn is not None) or (
#         rgb_alpha_fn is not None and alpha_fn is not None
#     )
#     # Either sigma functions or alpha functions must be provided, but not both
#     assert (sigma_fn is None or alpha_fn is None) and (
#         rgb_sigma_fn is None or rgb_alpha_fn is None
#     )
#
#     render_step_size = (far_plane - near_plane) / (num_samples_per_ray - 1)
#     ray_indices, t_starts, t_ends = grid_estimator.sampling(
#         rays_o,
#         rays_d,
#         alpha_fn=alpha_fn,
#         sigma_fn=sigma_fn,
#         near_plane=near_plane,
#         far_plane=far_plane,
#         render_step_size=render_step_size,
#         stratified=False,
#     )
#     (
#         rgb_map,
#         acc_map,
#         _,
#         extras,
#     ) = rendering(
#         t_starts,
#         t_ends,
#         ray_indices=ray_indices,
#         n_rays=n_rays,
#         rgb_alpha_fn=rgb_alpha_fn,
#         rgb_sigma_fn=rgb_sigma_fn,
#         render_bkgd=None,
#         chunk_size=shader_chunk,
#     )
#
#     return 1.0 - acc_map, rgb_map


def sample_volume_interaction(
    rays_o: Tensor,
    rays_d: Tensor,
    ray_indices: Tensor,
    t_starts: Tensor,
    t_ends: Tensor,
    n_rays: int,
    samples_per_pixel: int,
    transmittance_map: Tensor,
    extras: dict,
):
    """
    Sample interaction points for volumetric scattering, without re-evaluating density/weight/material/normal. We assume
    density/weight/material/normal/position of any point in an interval equals to the density/weight/material/normal/position
    of the midpoint of the interval.

    Args:
        rays_o (Tensor): Ray origins of shape (n_rays, 3).
        rays_d (Tensor): Normalized ray directions of shape (n_rays, 3).
        ray_indices (Tensor): Ray indices of the flattened samples. LongTensor with shape (all_samples).
        t_starts (Tensor): Per-sample start distance. Tensor with shape (all_samples,).
        t_ends (Tensor): Per-sample end distance. Tensor with shape (all_samples,).
        n_rays (int): Number of rays. Only useful when `ray_indices` is provided.
        samples_per_pixel (int): Number of samples per ray/pixel.
        transmittance_map (Tensor): Transmittance map of shape (n_rays, 1).
        extras (dict): Extra information for sample, including `weights`, `normals`, `albedo`, `roughness`, `metallic`.
            each of the value is a Tensor of shape (all_samples, ...).

    Returns:
        resampled_packed_info (Tensor): Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        resampled_ray_indices (Tensor): Resampled ray indices of shape (n_resamples,).
        resampled_weights (Tensor): Resampled weights of shape (n_resamples, 1).
        fg_indices (Tensor): Indices of foreground samples of shape (n_fg_resamples,).
        bg_indices (Tensor): Indices of background samples of shape (n_bg_resamples,).
        resampled_extras (dict): Resampled extra information for sample, including `positions`, `normals`, `albedo`,
            `roughness`, `metallic`, `t_dirs`. Each of the value is a Tensor of shape (n_resamples, ...).
    """

    # Importance sample points for volumetric scattering
    weights = extras["weights"]
    sdfs = extras["sdf"]
    with torch.no_grad():
        packed_info = pack_info(ray_indices, n_rays)
        (
            resampled_packed_info,
            resampled_midpoints,
            resampled_offsets,
            sampled_indices,
            sampled_fg_counts,
            sampled_bg_counts,
            surface_idx,
        ) = ray_resampling(
            packed_info,
            t_starts[..., None],
            t_ends[..., None],
            weights,
            sdfs,
            samples_per_pixel,
        )

        fg_indices = torch.nonzero(resampled_offsets < 1e4)[..., 0]
        bg_indices = torch.nonzero(resampled_offsets >= 1e4)[..., 0]
        resampled_ray_indices = unpack_info(
            resampled_packed_info, len(resampled_midpoints)
        )
        fg_resampled_ray_indices = resampled_ray_indices[fg_indices]
        bg_resampled_ray_indices = resampled_ray_indices[bg_indices]
        surface_idx = surface_idx[fg_resampled_ray_indices]
        # fg_resampled_midpoints = resampled_midpoints[fg_indices]
        # fg_resampled_offsets = resampled_offsets[fg_indices]
        fg_sampled_indices = sampled_indices[fg_indices]
        # bg_sampled_indices = sampled_indices[bg_indices]

    # Assemble resampled weights, positions, normals, albedo, roughness, metallic, t_dirs
    resampled_extras = {}
    if fg_sampled_indices.numel() > 0:
        resampled_weights = torch.zeros_like(resampled_midpoints[..., 0])
        # Weight of a resampled foreground point is the weight of the sampled
        # interval divided by the number of foreground re-samples in the interval
        resampled_fg_weights = (
            weights[fg_sampled_indices] / sampled_fg_counts[fg_sampled_indices].float()
        )
        # Weight of a resampled background point is the weight of the sampled
        # interval (here it is simply the transmittance of the ray) divided by
        # the number of background re-samples in the interval
        resampled_bg_weights = (
            transmittance_map[bg_resampled_ray_indices][..., 0]
            / sampled_bg_counts[bg_resampled_ray_indices].float()
        )
        resampled_weights.scatter_(0, fg_indices, resampled_fg_weights)
        resampled_weights.scatter_(0, bg_indices, resampled_bg_weights)

        t_dirs = rays_d[fg_resampled_ray_indices]
        # # For rays that we found zero-crossing point, we use the zero-crossing
        # # point as the resampled point for all samples. For rays that we did not
        # # find zero-crossing point, we use actual resampled points.
        # mask = surface_idx >= 0

        # t_coarse = ((t_starts + t_ends) / 2.0)
        # t = torch.zeros(
        #     (fg_sampled_indices.size(0),), device=t_coarse.device, dtype=t_coarse.dtype
        # )
        # t[mask] = t_coarse[surface_idx[mask]]
        # t[~mask] = t_coarse[fg_sampled_indices][~mask]

        # Ignore previous code-bloack. Zero-crossing on primary rays are now handled
        # inside the CUDA kernel.
        t = resampled_midpoints[fg_indices]

        # normals_coarse = extras["normals"]
        # normals = torch.zeros(
        #     fg_sampled_indices.size(0),
        #     3,
        #     device=normals_coarse.device,
        #     dtype=normals_coarse.dtype,
        # )
        # normals[mask] = normals_coarse[surface_idx[mask]]
        # normals[~mask] = normals_coarse[fg_sampled_indices][~mask]
        normals = extras["normals"][fg_sampled_indices]

        resampled_extras["sdf"] = sdfs[fg_sampled_indices]
        resampled_extras["alphas"] = extras["alphas"][fg_sampled_indices]
        resampled_extras["dists"] = (t_ends - t_starts)[..., None][fg_sampled_indices]
        resampled_extras["positions"] = (
            rays_o[fg_resampled_ray_indices]
            + rays_d[fg_resampled_ray_indices]
            * t
        )  # (n_fg_resamples, 3)

        resampled_extras["normals"] = normals
        resampled_extras["albedo"] = extras["albedo"][fg_sampled_indices]
        resampled_extras["roughness"] = extras["roughness"][fg_sampled_indices]
        resampled_extras["metallic"] = extras["metallic"][fg_sampled_indices]

        # (Inverse) incidence directions
        resampled_extras["t_dirs"] = t_dirs
    else:
        resampled_weights = torch.zeros((0, 1), device=rays_o.device)

        resampled_extras["sdf"] = torch.zeros((0,), device=rays_o.device)
        resampled_extras["alphas"] = torch.zeros((0,), device=rays_o.device)
        resampled_extras["dists"] = torch.zeros((0, 1), device=rays_o.device)
        resampled_extras["positions"] = torch.zeros((0, 3), device=rays_o.device)
        resampled_extras["normals"] = torch.zeros((0, 3), device=rays_o.device)
        resampled_extras["albedo"] = torch.zeros((0, 3), device=rays_o.device)
        resampled_extras["roughness"] = torch.zeros((0, 1), device=rays_o.device)
        resampled_extras["metallic"] = torch.zeros(
            (0, extras["metallic"].size(-1)), device=rays_o.device
        )
        resampled_extras["t_dirs"] = torch.zeros((0, 3), device=rays_o.device)

    return (
        resampled_packed_info,
        resampled_ray_indices,
        resampled_weights,
        fg_indices,
        bg_indices,
        resampled_extras,
    )


# def sample_volume_interaction_recompute(
#     rays_o: Tensor,
#     rays_d: Tensor,
#     ray_indices: Tensor,
#     t_starts: Tensor,
#     t_ends: Tensor,
#     n_rays: int,
#     samples_per_pixel: int,
#     transmittance_map: Tensor,
#     extras: dict,
#     normal_mats_alpha_fn: Callable,
# ):
#     """
#     Sample interaction points for volumetric scattering, and re-evaluate density/weight/material/normal of sampled points. We assume
#     density/weight/material/normal/position of any point in an interval equals to the density/weight/material/normal/position
#     of the midpoint of the interval.
#
#     Args:
#         rays_o (Tensor): Ray origins of shape (n_rays, 3).
#         rays_d (Tensor): Normalized ray directions of shape (n_rays, 3).
#         ray_indices (Tensor): Ray indices of the flattened samples. LongTensor with shape (all_samples).
#         t_starts (Tensor): Per-sample start distance. Tensor with shape (all_samples,).
#         t_ends (Tensor): Per-sample end distance. Tensor with shape (all_samples,).
#         n_rays (int): Number of rays. Only useful when `ray_indices` is provided.
#         samples_per_pixel (int): Number of samples per ray/pixel.
#         transmittance_map (Tensor): Transmittance map of shape (n_rays, 1).
#         extras (dict): Extra information for sample, including `weights`, `normals`, `albedo`, `roughness`, `metallic`.
#             each of the value is a Tensor of shape (all_samples, ...).
#         normal_mats_alpha_fn (Callable): A function that takes in samples {t_starts, t_ends,
#             ray indices} and returns the normal (..., 3), materials (..., 5/7), and alpha
#             values (...,). The shape `...` is the same as the shape of `t_starts`.
#
#     Returns:
#         resampled_packed_info (Tensor): Stores information on which samples belong to the same ray. \
#             See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
#         resampled_ray_indices (Tensor): Resampled ray indices of shape (n_resamples,).
#         resampled_weights (Tensor): Resampled weights of shape (n_resamples, 1).
#         resampled_opacities (Tensor): Resampled opacities of shape (n_resamples, 1).
#         resampled_extras (dict): Resampled extra information for sample, including `positions`, `normals`, `albedo`,
#             `roughness`, `metallic`, `t_dirs`. Each of the value is a Tensor of shape (n_resamples, ...).
#     """
#     # Importance sample points for volumetric scattering
#     alphas = extras["alphas"]
#     sdfs = extras["sdf"]
#     with torch.no_grad():
#         packed_info = pack_info(ray_indices, n_rays)
#         (
#             resampled_packed_info,
#             resampled_starts,
#             resampled_ends,
#             is_fg_sample,
#         ) = ray_resampling_sdf_fine(
#             packed_info,
#             t_starts[..., None],
#             t_ends[..., None],
#             alphas,
#             sdfs,
#             n_samples=samples_per_pixel,
#         )
#         resampled_ray_indices = unpack_info(
#             resampled_packed_info, len(resampled_starts)
#         )
#         resampled_starts = resampled_starts[is_fg_sample, 0]
#         resampled_ends = resampled_ends[is_fg_sample, 0]
#         resampled_ray_indices = resampled_ray_indices[is_fg_sample]
#
#     # Compute resampled weights, positions, normals, albedo, roughness, metallic, t_dirs
#     resampled_extras = {}
#     if is_fg_sample.any():
#         (
#             normals,
#             materials,
#             alphas,
#         ) = normal_mats_alpha_fn(resampled_starts, resampled_ends, resampled_ray_indices)
#
#         resampled_extras["positions"] = (
#             rays_o[resampled_ray_indices]
#             + rays_d[resampled_ray_indices]
#             * ((resampled_starts + resampled_ends)[..., None] / 2.0)
#         )  # (n_fg_resamples, 3)
#
#         resampled_extras["normals"] = normals
#         (
#             resampled_extras["albedo"],
#             resampled_extras["roughness"],
#             resampled_extras["metallic"],
#         ) = (
#             materials[..., :3],
#             materials[..., 3:4],
#             materials[..., 4:],
#         )
#
#         # (Inverse) incidence directions
#         resampled_extras["t_dirs"] = rays_d[resampled_ray_indices]
#
#         # Recompute rendering weights and opacities
#         resampled_weights, _ = render_weight_from_alpha(
#             alphas, ray_indices=resampled_ray_indices, n_rays=n_rays
#         )
#         resampled_opacities = accumulate_along_rays(
#             resampled_weights,
#             values=None,
#             ray_indices=resampled_ray_indices,
#             n_rays=n_rays,
#         )
#     else:
#         resampled_weights = torch.zeros((0, 1), device=rays_o.device)
#
#         resampled_extras["positions"] = torch.zeros((0, 3), device=rays_o.device)
#         resampled_extras["normals"] = torch.zeros((0, 3), device=rays_o.device)
#         resampled_extras["albedo"] = torch.zeros((0, 3), device=rays_o.device)
#         resampled_extras["roughness"] = torch.zeros((0, 1), device=rays_o.device)
#         resampled_extras["metallic"] = torch.zeros(
#             (0, extras["metallic"].size(-1)), device=rays_o.device
#         )
#         resampled_extras["t_dirs"] = torch.zeros((0, 3), device=rays_o.device)
#
#         resampled_weights = torch.zeros((0, 1), device=rays_o.device)
#         resampled_opacities = torch.zeros((0, 1), device=rays_o.device)
#
#     return (
#         resampled_packed_info,
#         resampled_ray_indices,
#         resampled_weights,
#         resampled_opacities,
#         resampled_extras,
#     )
