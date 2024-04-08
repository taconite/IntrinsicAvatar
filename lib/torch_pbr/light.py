import numpy as np
import torch
import torch.nn.functional as F
import tinycudann as tcnn

# import models

from omegaconf import OmegaConf
from .utils import nvdiffrecmc_util as util
from .utils.warp_utils import sample_uniform_sphere
from .utils.light_utils import (
    compute_energy,
    fibonacci_sphere,
    eval_SGs,
    xyz2lonlat,
    lonlat2uv,
    uv2lonlat,
    lonlat2xyz,
    xyz2lonlat_blender,
    lonlat2uv_blender,
    uv2lonlat_blender,
    lonlat2xyz_blender,
)

######################################################################################
# Monte-carlo sampled environment light with PDF / CDF computation
######################################################################################


class EnvironmentLightBase(torch.nn.Module):
    def __init__(self, config):
        if config.xyz2lonlat_mode == "blender":
            self.xyz2lonlat = xyz2lonlat_blender
            self.lonlat2uv = lonlat2uv_blender
            self.uv2lonlat = uv2lonlat_blender
            self.lonlat2xyz = lonlat2xyz_blender
            self.sin_func = lambda theta: torch.sin(theta)
        else:
            self.xyz2lonlat = xyz2lonlat
            self.lonlat2uv = lonlat2uv
            self.uv2lonlat = uv2lonlat
            self.lonlat2xyz = lonlat2xyz
            self.sin_func = lambda theta: torch.sin(np.pi / 2.0 - theta)
        super(EnvironmentLightBase, self).__init__()

    # def clone(self):
    #     raise NotImplementedError("EnvironmentLightBase is an abstract class")

    # def clamp_(self, min=None, max=None):
    #     raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def pdf(self, directions):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    def eval(self, directions):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def sample(self, num_samples: int):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def update_pdf(self):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def generate_image(self):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def sample_equirectangular_stratified(
        self,
        batch_size: int,
        n_rows: int,
        n_cols: int,
        device: torch.device,
    ):
        """
        Sampling of the environment map with potentially sample stratification
        during training. Modified from TensoIR's implementation.  Note that here
        we uniformly sample pixels from the equirectangular environment map,
        this is NOT equivalent to uniformly sampling directions on the sphere,
        thus the returned inverse PDFs will have small values at the poles.
        Args:
            batch_size: The number of batches (pixels) to sample
            n_rows: The number of rows for each batch
            n_cols: The number of columns for each batch
            device: The device to put the sampled directions on
            shuffle: Whether to shuffle the batches
        Returns:
            directions: A tensor of shape (batch_size * n_rows * n_cols, 3)
                        containing sampled directions
            inv_pdf: A tensor of shape (batch_size * n_rows * n_cols, 1)
                     containing the inverse PDFs of the sampled directions
            inv_pdf_normalized: A tensor of shape (batch_size * n_rows * n_cols,
                    1) containing the normalized inverse PDFs of the sampled directions
        """
        lat_step_size = np.pi / n_rows
        lng_step_size = 2 * np.pi / n_cols

        # Generate theta in [pi/2, -pi/2] and phi in [pi, -pi]
        theta, phi = torch.meshgrid(
            [
                torch.linspace(
                    np.pi / 2 - 0.5 * lat_step_size,
                    -np.pi / 2 + 0.5 * lat_step_size,
                    n_rows,
                    device=device,
                ),
                torch.linspace(
                    np.pi - 0.5 * lng_step_size,
                    -np.pi + 0.5 * lng_step_size,
                    n_cols,
                    device=device,
                ),
            ],
            indexing="ij",
        )
        sin_theta = self.sin_func(theta)
        inv_pdf = 4 * torch.pi * sin_theta  # [H, W]    # no normalization - normalization is handled
                                                        # by volumetric rendering weights
        # inv_pdf_normalized = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [H, W]
        inv_pdf_vis = n_rows * n_cols * sin_theta / torch.sum(sin_theta)  # [H, W]
        inv_pdf = inv_pdf.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, H, W]
        # inv_pdf_normalized = (
        #     inv_pdf_normalized.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, H, W]
        # )
        inv_pdf_vis = inv_pdf_vis.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, H, W]
        if self.training:
            phi_jitter = lng_step_size * (
                torch.rand(batch_size, n_rows, n_cols, device=device) - 0.5
            )
            theta_jitter = lat_step_size * (
                torch.rand(batch_size, n_rows, n_cols, device=device) - 0.5
            )

            theta, phi = theta[None, ...] + theta_jitter, phi[None, ...] + phi_jitter

        # directions = torch.stack(
        #     [
        #         torch.cos(phi) * torch.cos(theta),
        #         torch.sin(phi) * torch.cos(theta),
        #         torch.sin(theta),
        #     ],
        #     dim=-1,
        # )  # training: [B, H, W, 3], testing: [H, W, 3]
        directions = self.lonlat2xyz(torch.stack([phi, theta], dim=-1))
        directions = F.normalize(directions, dim=-1)
        if not self.training:
            directions = directions.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Final return: [B*H*W, 3], [B*H*W, 1], [B*H*W, 1]
        return (
            directions.reshape(-1, 3),
            inv_pdf.reshape(-1, 1),
            # inv_pdf_vis.reshape(-1, 1),
        )

    @torch.no_grad()
    def sample_uniform_sphere_stratified(
        self,
        batch_size: int,
        n_rows: int,
        n_cols: int,
        device: torch.device,
    ):
        """
        Uniform sampling of the environment map with potentially sample
        stratification during training.
        Args:
            batch_size: The number of batches (pixels) to sample
            n_rows: The number of rows for each batch
            n_cols: The number of columns for each batch
            device: The device to put the sampled directions on
            shuffle: Whether to shuffle the batches
        Returns:
            directions: A tensor of shape (batch_size * n_rows * n_cols, 3)
                        containing sampled directions
            inv_pdf: A tensor of shape (batch_size * n_rows * n_cols, 1)
                     containing the inverse PDFs of the sampled directions
            inv_pdf_normalized: A tensor of shape (batch_size * n_rows * n_cols,
                    1) containing the normalized inverse PDFs of the sampled directions
        """
        # Evaluate envmap at the pixel centers
        v, u = torch.meshgrid(
            (torch.arange(0, n_rows, dtype=torch.float32, device=device) + 0.5),
            (torch.arange(0, n_cols, dtype=torch.float32, device=device) + 0.5),
        )

        # Add jitter during training
        if self.training:
            u = u[None, ...] + (torch.rand(batch_size, n_rows, n_cols, device=device) - 0.5)
            v = v[None, ...] + (torch.rand(batch_size, n_rows, n_cols, device=device) - 0.5)

        u = u / n_cols
        v = v / n_rows

        assert (u >= 0).all() and (u <= 1).all()
        assert (v >= 0).all() and (v <= 1).all()

        # Warp random samples onto a unit sphere
        # we use [v, u] instead of [u, v] becaue `sample_uniform_sphere` will
        # map the first column to z-axis
        directions = sample_uniform_sphere(
            torch.stack([v, u], dim=-1).reshape(-1, 2)
        ).reshape(-1, 3)
        directions = F.normalize(directions, dim=-1)

        if not self.training:
            directions = directions.repeat(batch_size, 1)

        inv_pdf = (
            torch.ones_like(directions[..., :1]) * 4 * torch.pi
        )  # no normalization - normalization is handled by volumetric rendering weights

        return directions, inv_pdf.reshape(-1, 1)


# @models.register("envlight-tensor")
class EnvironmentLightTensor(EnvironmentLightBase):
    def __init__(self, config):
        super(EnvironmentLightTensor, self).__init__(config)
        # self.mtx = None
        scale = config.envlight_config.scale
        bias = config.envlight_config.bias
        base_res = config.envlight_config.base_res
        if config.envlight_config.hdr_filepath is None:
            base = (
                torch.rand(base_res, base_res, 3, dtype=torch.float32, device="cuda")
                * scale
                + bias
            )
        else:
            base = (
                torch.tensor(
                    util.load_image(config.envlight_config.hdr_filepath),
                    dtype=torch.float32,
                    device="cuda",
                )
                * scale
                + bias
            )

        self.register_parameter("base", torch.nn.Parameter(base))

        self.pdf_scale = (self.base.shape[0] * self.base.shape[1]) / (2 * np.pi * np.pi)
        self.update_pdf()

    # def xfm(self, mtx):
    #     self.mtx = mtx

    def parameters(self):
        return [self.base]

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    @torch.no_grad()
    def pdf(self, directions):
        """
        Compute the PDFs of the given directions based on the environment map
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N,) containing the PDFs for each input direction
        """
        # Convert the 3D directions to 2D indices in the environment map
        # phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        # theta = torch.acos(directions[:, 2])  # Compute elevation angle
        # u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        # v = theta / np.pi  # Map elevation to [0, 1]
        lonlat = self.xyz2lonlat(directions)
        _, theta = lonlat[:, 0], lonlat[:, 1]
        uv = self.lonlat2uv(lonlat)
        u, v = uv[:, 0], uv[:, 1]

        # Convert u, v to discrete indices
        col_indices = torch.clamp(
            torch.floor(u * (self.cols.shape[1] - 1)), min=0, max=self.cols.shape[1] - 2
        )
        row_indices = torch.clamp(
            torch.floor(v * (self.rows.shape[0] - 1)), min=0, max=self.rows.shape[0] - 2
        )

        # Get PDF values at the indices
        sin_theta = self.sin_func(theta)
        pdf_values = torch.where(
            sin_theta > 0,
            self._pdf[row_indices.long(), col_indices.long()]
            * self.pdf_scale
            / sin_theta,
            torch.zeros_like(sin_theta),
        )

        return pdf_values.unsqueeze(-1)

    def eval(self, directions):
        """
        Evaluate the environment light intensities at the given directions
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N, C) containing the environment light intensities at the input directions
        """
        # Convert the 3D directions to 2D indices in the environment map
        # Assume the azimuth (phi) is in [-pi, pi] and the elevation (theta) is in [0, pi]
        # phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        # theta = torch.acos(directions[:, 2])  # Compute elevation angle
        # u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        # v = theta / np.pi  # Map elevation to [0, 1]
        lonlat = self.xyz2lonlat(directions)
        uv = self.lonlat2uv(lonlat)
        u, v = uv[:, 0], uv[:, 1]
        assert (u >= 0).all() and (u <= 1).all()
        assert (v >= 0).all() and (v <= 1).all()

        # Create a grid for grid_sample. The grid values should be in the range of [-1, 1]
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).reshape(1, 1, -1, 2)

        # Add a batch dimension to the environment map for grid_sample
        base_batch = self.base.unsqueeze(0).permute(0, 3, 1, 2)

        # Use grid_sample for interpolation. The function assumes the grid values to be in the range of [-1, 1]
        intensity = torch.nn.functional.grid_sample(
            base_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Squeeze the batch dimension and transpose the result to match the input shape
        intensity = intensity.reshape(self.base.shape[-1], -1).transpose(0, 1)

        if self.training:
            return F.softplus(intensity, beta=100)
        else:
            return intensity

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        Importance sample continuous locations on the environment light based on discrete CDFs
        Args:
            num_samples: Number of samples to generate
        Returns:
            A tuple (indices, pdfs) where:
                indices: A tensor of shape (num_samples, 2) containing sampled row and column indices
                pdfs: A tensor of shape (num_samples,) containing the pdf values of the sampled indices
        """
        # Generate random numbers for rows and columns
        u1 = torch.rand(num_samples, device=self.base.device)
        u2 = (
            torch.rand(num_samples, device=self.base.device).reshape(-1, 1).contiguous()
        )

        # Find the row indices based on the random numbers u1 and the row CDF
        # TODO: check for divide-by-zero cases - probably not needed
        row_indices = torch.searchsorted(self.rows[:, 0].contiguous(), u1, right=True)
        below = torch.max(torch.zeros_like(row_indices - 1), row_indices - 1)
        above = torch.min(
            (self.rows.shape[0] - 1) * torch.ones_like(row_indices), row_indices
        )
        row_fracs = (u1 - self.rows[below, 0]) / (
            self.rows[above, 0] - self.rows[below, 0]
        )
        row_indices = below

        # For each row index, find the column index based on the random numbers u2 and the column CDF
        # Use advanced indexing to vectorize the operation
        col_indices = torch.searchsorted(
            self.cols[row_indices], u2, right=True
        ).squeeze(-1)
        below = torch.max(torch.zeros_like(col_indices - 1), col_indices - 1)
        above = torch.min(
            (self.cols.shape[-1] - 1) * torch.ones_like(col_indices), col_indices
        )
        col_fracs = (u2.squeeze(-1) - self.cols[row_indices, below]) / (
            self.cols[row_indices, above] - self.cols[row_indices, below]
        )
        col_indices = below

        # Concatenate the row and column indices to get a 2D index for each sample
        # Add the fractions to get continuous coordinates
        uv = torch.stack(
            [
                (col_indices + col_fracs) / self.base.shape[1],
                (row_indices + row_fracs) / self.base.shape[0],
            ],
            dim=1,
        )

        # Convert the 2D indices to spherical coordinates
        # theta = uv[:, 1] * np.pi
        # phi = uv[:, 0] * np.pi * 2 - np.pi
        lonlat = self.uv2lonlat(uv)

        # Convert spherical coordinates to directions
        # sin_theta = torch.sin(theta)
        # cos_theta = torch.cos(theta)
        # sin_phi = torch.sin(phi)
        # cos_phi = torch.cos(phi)

        # directions = torch.stack(
        #     [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=1
        # )
        directions = self.lonlat2xyz(lonlat)
        directions = F.normalize(directions, dim=1)

        return directions

    @torch.no_grad()
    def update_pdf(self):
        # Compute PDF
        Y = util.pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
        self._pdf = torch.max(self.base, dim=-1)[0] * torch.sin(
            Y * np.pi
        )  # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
        self._pdf[self._pdf <= 0] = 1e-6  # avoid divide by zero in sample()
        self._pdf = self._pdf / torch.sum(self._pdf)  # discrete pdf

        # Compute cumulative sums over the columns and rows
        self.cols = torch.cumsum(self._pdf, dim=1)
        self.rows = torch.cumsum(
            self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0
        )

        # Normalize
        # TODO: for columns/rows with all 0s, use uniform distribution
        self.cols = self.cols / torch.where(
            self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols)
        )
        self.rows = self.rows / torch.where(
            self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows)
        )

        # Prepend 0s to all CDFs
        self.cols = torch.cat([torch.zeros_like(self.cols[:, :1]), self.cols], dim=1)
        self.rows = torch.cat([torch.zeros_like(self.rows[:1, :]), self.rows], dim=0)

        # self._pdf *= self.base.shape[1] * self.base.shape[0]

    @torch.no_grad()
    def generate_image(self):
        return self.base.detach().cpu().numpy()


# TODO: there should be a way to analytically derived the sample() and pdf() functions
# @models.register("envlight-SG")
class EnvironmentLightSG(EnvironmentLightBase):
    def __init__(self, config):
        super(EnvironmentLightSG, self).__init__(config)
        self.num_SGs = config.envlight_config.num_SGs
        if isinstance(config.envlight_config.base_res, int):
            self.base_res = (
                config.envlight_config.base_res,
                config.envlight_config.base_res,
            )
        else:
            self.base_res = config.envlight_config.base_res

        lgtSGs = torch.randn(
            self.num_SGs, 7, device="cuda"
        )  # [M, 7]; lobe + lambda + mu
        lgtSGs[:, -2:] = lgtSGs[:, -3:-2].expand((-1, 2))

        # make sure lambda is not too close to zero
        lgtSGs[:, 3:4] = 10.0 + torch.abs(lgtSGs[:, 3:4] * 20.0)
        # init envmap energy
        energy = compute_energy(lgtSGs)
        lgtSGs[:, 4:] = (
            torch.abs(lgtSGs[:, 4:])
            / torch.sum(energy, dim=0, keepdim=True)
            * 2.0
            * np.pi
            * 0.8
        )
        # energy = compute_energy(lgtSGs)

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.num_SGs // 2).astype(np.float32)
        lgtSGs[: self.num_SGs // 2, :3] = torch.from_numpy(lobes)
        lgtSGs[self.num_SGs // 2 :, :3] = torch.from_numpy(lobes)

        self.register_parameter("lgtSGs", torch.nn.Parameter(lgtSGs))

        # self.pdf_scale = (self.base_res[0] * self.base_res[1]) / (2 * np.pi * np.pi)
        # self.update_pdf()

    # def xfm(self, mtx):
    #     self.mtx = mtx

    def parameters(self):
        return [self.lgtSGs]

    # def clone(self):
    #     return EnvironmentLight(self.base.clone().detach())

    # def clamp_(self, min=None, max=None):
    #     self.base.clamp_(min, max)

    @torch.no_grad()
    def pdf(self, directions):
        """
        Compute the PDFs of the given directions based on the environment map
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N,) containing the PDFs for each input direction
        """
        # TODO: implement precise SG pdf https://arxiv.org/pdf/2303.16617.pdf
        raise NotImplementedError(
            "pdf() function of EnvironmentLightSG is not implemented"
        )
        # Convert the 3D directions to 2D indices in the environment map
        # phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        # theta = torch.acos(directions[:, 2])  # Compute elevation angle
        # u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        # v = theta / np.pi  # Map elevation to [0, 1]
        lonlat = self.xyz2lonlat(directions)
        _, theta = lonlat[:, 0], lonlat[:, 1]
        uv = self.lonlat2uv(lonlat)
        u, v = uv[:, 0], uv[:, 1]

        # Convert u, v to discrete indices
        col_indices = torch.clamp(
            torch.floor(u * (self.cols.shape[1] - 1)), min=0, max=self.cols.shape[1] - 2
        )
        row_indices = torch.clamp(
            torch.floor(v * (self.rows.shape[0] - 1)), min=0, max=self.rows.shape[0] - 2
        )

        # Get PDF values at the indices
        sin_theta = self.sin_func(theta)
        pdf_values = torch.where(
            sin_theta > 0,
            self._pdf[row_indices.long(), col_indices.long()]
            * self.pdf_scale
            / sin_theta,
            torch.zeros_like(sin_theta),
        )

        return pdf_values.unsqueeze(-1)

    def eval(self, directions):
        """
        Evaluate the environment light intensities at the given directions
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N, C) containing the environment light intensities at the input directions
        """
        return eval_SGs(self.lgtSGs, directions)

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        Importance sample continuous locations on the environment light based on discrete CDFs
        Args:
            num_samples: Number of samples to generate
        Returns:
            A tuple (indices, pdfs) where:
                indices: A tensor of shape (num_samples, 2) containing sampled row and column indices
                pdfs: A tensor of shape (num_samples,) containing the pdf values of the sampled indices
        """
        # TODO: implement precise SG sampling https://arxiv.org/pdf/2303.16617.pdf
        raise NotImplementedError(
            "sample() function of EnvironmentLightSG is not implemented"
        )
        # Generate random numbers for rows and columns
        u1 = torch.rand(num_samples, device=self.lgtSGs.device)
        u2 = (
            torch.rand(num_samples, device=self.lgtSGs.device)
            .reshape(-1, 1)
            .contiguous()
        )

        # Find the row indices based on the random numbers u1 and the row CDF
        # TODO: check for divide-by-zero cases - probably not needed
        row_indices = torch.searchsorted(self.rows[:, 0].contiguous(), u1, right=True)
        below = torch.max(torch.zeros_like(row_indices - 1), row_indices - 1)
        above = torch.min(
            (self.rows.shape[0] - 1) * torch.ones_like(row_indices), row_indices
        )
        row_fracs = (u1 - self.rows[below, 0]) / (
            self.rows[above, 0] - self.rows[below, 0]
        )
        row_indices = below

        # For each row index, find the column index based on the random numbers u2 and the column CDF
        # Use advanced indexing to vectorize the operation
        col_indices = torch.searchsorted(
            self.cols[row_indices], u2, right=True
        ).squeeze(-1)
        below = torch.max(torch.zeros_like(col_indices - 1), col_indices - 1)
        above = torch.min(
            (self.cols.shape[-1] - 1) * torch.ones_like(col_indices), col_indices
        )
        col_fracs = (u2.squeeze(-1) - self.cols[row_indices, below]) / (
            self.cols[row_indices, above] - self.cols[row_indices, below]
        )
        col_indices = below

        # Concatenate the row and column indices to get a 2D index for each sample
        # Add the fractions to get continuous coordinates
        uv = torch.stack(
            [
                (col_indices + col_fracs) / self.base_res[1],
                (row_indices + row_fracs) / self.base_res[0],
            ],
            dim=1,
        )

        # Convert the 2D indices to spherical coordinates
        # theta = uv[:, 1] * np.pi
        # phi = uv[:, 0] * np.pi * 2 - np.pi
        lonlat = self.uv2lonlat(uv)

        # Convert spherical coordinates to directions
        # sin_theta = torch.sin(theta)
        # cos_theta = torch.cos(theta)
        # sin_phi = torch.sin(phi)
        # cos_phi = torch.cos(phi)

        # directions = torch.stack(
        #     [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=1
        # )
        directions = self.lonlat2xyz(lonlat)
        directions = F.normalize(directions, dim=1)

        return directions

    @torch.no_grad()
    def update_pdf(self):
        # TODO: implement precise SG pdf https://arxiv.org/pdf/2303.16617.pdf
        raise NotImplementedError(
            "sample() function of EnvironmentLightSG is not implemented"
        )
        # Evaluate SGs at the pixel centers
        XY = util.pixel_grid(self.base_res[1], self.base_res[0])

        # Convert the (u, v) \in [0, 1]^2 to spherical coordinates
        theta = XY[..., 1] * np.pi
        phi = XY[..., 0] * np.pi * 2 - np.pi

        # Convert spherical coordinates to 3D direction vectors
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        directions = torch.stack(
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=-1
        )
        directions = F.normalize(directions.reshape(-1, 3), dim=1)
        # Compute envmap from SGs
        envmap = eval_SGs(self.lgtSGs, directions).reshape(
            self.base_res[0], self.base_res[1], -1
        )
        # Compute PDF
        Y = XY[..., 1]
        self._pdf = torch.max(envmap, dim=-1)[0] * torch.sin(
            Y * np.pi
        )  # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
        self._pdf[self._pdf <= 0] = 1e-6  # avoid divide by zero in sample()
        self._pdf = self._pdf / torch.sum(self._pdf)  # discrete pdf

        # Compute cumulative sums over the columns and rows
        self.cols = torch.cumsum(self._pdf, dim=1)
        self.rows = torch.cumsum(
            self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0
        )

        # Normalize
        # TODO: for columns/rows with all 0s, use uniform distribution
        self.cols = self.cols / torch.where(
            self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols)
        )
        self.rows = self.rows / torch.where(
            self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows)
        )

        # Prepend 0s to all CDFs
        self.cols = torch.cat([torch.zeros_like(self.cols[:, :1]), self.cols], dim=1)
        self.rows = torch.cat([torch.zeros_like(self.rows[:1, :]), self.rows], dim=0)

    @torch.no_grad()
    def generate_image(self):
        # Just for visualization, we use the spherical coordinate instead of
        # geographical coordinates
        # Evaluate SGs at the pixel centers
        XY = util.pixel_grid(self.base_res[1], self.base_res[0])

        # Convert the (u, v) \in [0, 1]^2 to spherical coordinates
        theta = XY[..., 1] * np.pi
        phi = XY[..., 0] * np.pi * 2 - np.pi

        # Convert spherical coordinates to 3D direction vectors
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        directions = torch.stack(
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=-1
        )
        directions = F.normalize(directions.reshape(-1, 3), dim=1)
        # Compute envmap from SGs
        envmap = eval_SGs(self.lgtSGs, directions).reshape(
            self.base_res[0], self.base_res[1], -1
        )

        return envmap.detach().cpu().numpy()


# @models.register("envlight-mlp")
class EnvironmentLightMLP(EnvironmentLightBase):
    def __init__(self, config):
        super(EnvironmentLightMLP, self).__init__(config)
        if isinstance(config.envlight_config.base_res, int):
            self.base_res = (
                config.envlight_config.base_res,
                config.envlight_config.base_res,
            )
        else:
            self.base_res = config.envlight_config.base_res

        self.dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.net = tcnn.Network(
            n_input_dims=self.dir_encoder.n_output_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Softplus",
                "n_neurons": 128,
                "n_hidden_layers": 3,
            },
        )

        self.pdf_scale = (self.base_res[0] * self.base_res[1]) / (2 * np.pi * np.pi)
        self.update_pdf()

    @torch.no_grad()
    def pdf(self, directions):
        """
        Compute the PDFs of the given directions based on the environment map
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N,) containing the PDFs for each input direction
        """
        # Convert the 3D directions to 2D indices in the environment map
        phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        theta = torch.acos(directions[:, 2])  # Compute elevation angle
        u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        v = theta / np.pi  # Map elevation to [0, 1]

        # Convert u, v to discrete indices
        col_indices = torch.clamp(
            torch.floor(u * (self.cols.shape[1] - 1)), min=0, max=self.cols.shape[1] - 2
        )
        row_indices = torch.clamp(
            torch.floor(v * (self.rows.shape[0] - 1)), min=0, max=self.rows.shape[0] - 2
        )

        # Get PDF values at the indices
        sin_theta = self.sin_func(theta)
        pdf_values = torch.where(
            sin_theta > 0,
            self._pdf[row_indices.long(), col_indices.long()]
            * self.pdf_scale
            / sin_theta,
            torch.zeros_like(sin_theta),
        )

        return pdf_values.unsqueeze(-1)

    def eval(self, directions):
        """
        Evaluate the environment light intensities at the given directions
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N, C) containing the environment light intensities at the input directions
        """
        d = (directions + 1.) / 2.  # (-1, 1) => (0, 1)
        d = self.dir_encoder(d)
        intensity = self.net(d).float()
        return intensity

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        Importance sample continuous locations on the environment light based on discrete CDFs
        Args:
            num_samples: Number of samples to generate
        Returns:
            A tuple (indices, pdfs) where:
                indices: A tensor of shape (num_samples, 2) containing sampled row and column indices
                pdfs: A tensor of shape (num_samples,) containing the pdf values of the sampled indices
        """
        # Generate random numbers for rows and columns
        u1 = torch.rand(num_samples, device=self.rows.device)
        u2 = (
            torch.rand(num_samples, device=self.rows.device)
            .reshape(-1, 1)
            .contiguous()
        )

        # Find the row indices based on the random numbers u1 and the row CDF
        # TODO: check for divide-by-zero cases - probably not needed
        row_indices = torch.searchsorted(self.rows[:, 0].contiguous(), u1, right=True)
        below = torch.max(torch.zeros_like(row_indices - 1), row_indices - 1)
        above = torch.min(
            (self.rows.shape[0] - 1) * torch.ones_like(row_indices), row_indices
        )
        row_fracs = (u1 - self.rows[below, 0]) / (
            self.rows[above, 0] - self.rows[below, 0]
        )
        row_indices = below

        # For each row index, find the column index based on the random numbers u2 and the column CDF
        # Use advanced indexing to vectorize the operation
        col_indices = torch.searchsorted(
            self.cols[row_indices], u2, right=True
        ).squeeze(-1)
        below = torch.max(torch.zeros_like(col_indices - 1), col_indices - 1)
        above = torch.min(
            (self.cols.shape[-1] - 1) * torch.ones_like(col_indices), col_indices
        )
        col_fracs = (u2.squeeze(-1) - self.cols[row_indices, below]) / (
            self.cols[row_indices, above] - self.cols[row_indices, below]
        )
        col_indices = below

        # Concatenate the row and column indices to get a 2D index for each sample
        # Add the fractions to get continuous coordinates
        uv = torch.stack(
            [
                (col_indices + col_fracs) / self.base_res[1],
                (row_indices + row_fracs) / self.base_res[0],
            ],
            dim=1,
        )

        # Convert the 2D indices to spherical coordinates
        theta = uv[:, 1] * np.pi
        phi = uv[:, 0] * np.pi * 2 - np.pi

        # Convert spherical coordinates to directions
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        directions = torch.stack(
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=1
        )
        directions = F.normalize(directions, dim=1)

        return directions

    @torch.no_grad()
    def update_pdf(self):
        # Evaluate the MLP at the pixel centers
        XY = util.pixel_grid(self.base_res[1], self.base_res[0])

        # Convert the (u, v) \in [0, 1]^2 to spherical coordinates
        theta = XY[..., 1] * np.pi
        phi = XY[..., 0] * np.pi * 2 - np.pi

        # Convert spherical coordinates to 3D direction vectors
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        directions = torch.stack(
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=-1
        )
        directions = F.normalize(directions.reshape(-1, 3), dim=1)
        # Compute envmap from the MLP
        envmap = self.eval(directions).reshape(
            self.base_res[0], self.base_res[1], -1
        )
        # Compute PDF
        Y = XY[..., 1]
        self._pdf = torch.max(envmap, dim=-1)[0] * torch.sin(
            Y * np.pi
        )  # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
        self._pdf[self._pdf <= 0] = 1e-6  # avoid divide by zero in sample()
        self._pdf = self._pdf / torch.sum(self._pdf)  # discrete pdf

        # Compute cumulative sums over the columns and rows
        self.cols = torch.cumsum(self._pdf, dim=1)
        self.rows = torch.cumsum(
            self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0
        )

        # Normalize
        # TODO: for columns/rows with all 0s, use uniform distribution
        self.cols = self.cols / torch.where(
            self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols)
        )
        self.rows = self.rows / torch.where(
            self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows)
        )

        # Prepend 0s to all CDFs
        self.cols = torch.cat([torch.zeros_like(self.cols[:, :1]), self.cols], dim=1)
        self.rows = torch.cat([torch.zeros_like(self.rows[:1, :]), self.rows], dim=0)

    @torch.no_grad()
    def generate_image(self):
        # Evaluate the MLP at the pixel centers
        XY = util.pixel_grid(self.base_res[1], self.base_res[0])

        # Convert the (u, v) \in [0, 1]^2 to spherical coordinates
        theta = XY[..., 1] * np.pi
        phi = XY[..., 0] * np.pi * 2 - np.pi

        # Convert spherical coordinates to 3D direction vectors
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        directions = torch.stack(
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=-1
        )
        directions = F.normalize(directions.reshape(-1, 3), dim=1)
        # Compute envmap from the MLP
        envmap = self.eval(directions).reshape(
            self.base_res[0], self.base_res[1], -1
        )

        return envmap.detach().cpu().numpy()


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


# @models.register("envlight-mlp")
class EnvironmentLightNGP(EnvironmentLightBase):
    def __init__(self, config):
        super(EnvironmentLightNGP, self).__init__(config)
        if isinstance(config.envlight_config.base_res, int):
            self.base_res = (
                config.envlight_config.base_res,
                config.envlight_config.base_res,
            )
        else:
            self.base_res = config.envlight_config.base_res

        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=3,
            encoding_config=config_to_primitive(config.envlight_config.encoding_config),
            network_config=config_to_primitive(
                config.envlight_config.mlp_network_config
            ),
        )

        self.pdf_scale = (self.base_res[0] * self.base_res[1]) / (2 * np.pi * np.pi)
        self.update_pdf()

    @torch.no_grad()
    def pdf(self, directions):
        """
        Compute the PDFs of the given directions based on the environment map
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N,) containing the PDFs for each input direction
        """
        # Convert the 3D directions to 2D indices in the environment map
        # phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        # theta = torch.acos(directions[:, 2])  # Compute elevation angle
        # u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        # v = theta / np.pi  # Map elevation to [0, 1]
        lonlat = self.xyz2lonlat(directions)
        _, theta = lonlat[:, 0], lonlat[:, 1]
        uv = self.lonlat2uv(lonlat)
        u, v = uv[:, 0], uv[:, 1]

        # Convert u, v to discrete indices
        col_indices = torch.clamp(
            torch.floor(u * (self.cols.shape[1] - 1)), min=0, max=self.cols.shape[1] - 2
        )
        row_indices = torch.clamp(
            torch.floor(v * (self.rows.shape[0] - 1)), min=0, max=self.rows.shape[0] - 2
        )

        # Get PDF values at the indices
        sin_theta = self.sin_func(theta)
        pdf_values = torch.where(
            sin_theta > 0,
            self._pdf[row_indices.long(), col_indices.long()]
            * self.pdf_scale
            / sin_theta,
            torch.zeros_like(sin_theta),
        )

        return pdf_values.unsqueeze(-1)

    def eval_uv(self, uv):
        """
        Evaluate the environment light intensities at the given uv coordinates
        Args:
            uv: A tensor of shape (N, 2) containing uv coordinates
        Returns:
            A tensor of shape (N, C) containing the environment light intensities at the input directions
        """
        assert (uv >= 0).all() and (uv <= 1).all()

        return self.net(uv).float()

    def eval(self, directions):
        """
        Evaluate the environment light intensities at the given directions
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N, C) containing the environment light intensities at the input directions
        """
        # Convert the 3D directions to 2D indices in the environment map
        # Assume the azimuth (phi) is in [-pi, pi] and the elevation (theta) is in [0, pi]
        # phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        # theta = torch.acos(directions[:, 2])  # Compute elevation angle
        # u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        # v = theta / np.pi  # Map elevation to [0, 1]
        lonlat = self.xyz2lonlat(directions)
        uv = self.lonlat2uv(lonlat)
        u, v = uv[:, 0], uv[:, 1]
        assert (u >= 0).all() and (u <= 1).all()
        assert (v >= 0).all() and (v <= 1).all()

        # Create a grid for grid_sample. The grid values should be in the range of [0, 1]
        uv = torch.stack([u, v], dim=-1)

        return self.net(uv).float()

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        Importance sample continuous locations on the environment light based on discrete CDFs
        Args:
            num_samples: Number of samples to generate
        Returns:
            A tuple (indices, pdfs) where:
                indices: A tensor of shape (num_samples, 2) containing sampled row and column indices
                pdfs: A tensor of shape (num_samples,) containing the pdf values of the sampled indices
        """
        # Generate random numbers for rows and columns
        u1 = torch.rand(num_samples, device=self.rows.device)
        u2 = (
            torch.rand(num_samples, device=self.rows.device)
            .reshape(-1, 1)
            .contiguous()
        )

        # Find the row indices based on the random numbers u1 and the row CDF
        # TODO: check for divide-by-zero cases - probably not needed
        row_indices = torch.searchsorted(self.rows[:, 0].contiguous(), u1, right=True)
        below = torch.max(torch.zeros_like(row_indices - 1), row_indices - 1)
        above = torch.min(
            (self.rows.shape[0] - 1) * torch.ones_like(row_indices), row_indices
        )
        row_fracs = (u1 - self.rows[below, 0]) / (
            self.rows[above, 0] - self.rows[below, 0]
        )
        row_indices = below

        # For each row index, find the column index based on the random numbers u2 and the column CDF
        # Use advanced indexing to vectorize the operation
        col_indices = torch.searchsorted(
            self.cols[row_indices], u2, right=True
        ).squeeze(-1)
        below = torch.max(torch.zeros_like(col_indices - 1), col_indices - 1)
        above = torch.min(
            (self.cols.shape[-1] - 1) * torch.ones_like(col_indices), col_indices
        )
        col_fracs = (u2.squeeze(-1) - self.cols[row_indices, below]) / (
            self.cols[row_indices, above] - self.cols[row_indices, below]
        )
        col_indices = below

        # Concatenate the row and column indices to get a 2D index for each sample
        # Add the fractions to get continuous coordinates
        uv = torch.stack(
            [
                (col_indices + col_fracs) / self.base_res[1],
                (row_indices + row_fracs) / self.base_res[0],
            ],
            dim=1,
        )

        # Convert the 2D indices to spherical coordinates
        # theta = uv[:, 1] * np.pi
        # phi = uv[:, 0] * np.pi * 2 - np.pi
        lonlat = self.uv2lonlat(uv)

        # Convert spherical coordinates to directions
        # sin_theta = torch.sin(theta)
        # cos_theta = torch.cos(theta)
        # sin_phi = torch.sin(phi)
        # cos_phi = torch.cos(phi)

        # directions = torch.stack(
        #     [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=1
        # )
        directions = self.lonlat2xyz(lonlat)
        directions = F.normalize(directions, dim=1)

        return directions

    @torch.no_grad()
    def update_pdf(self):
        # Evaluate the MLP at the pixel centers
        XY = util.pixel_grid(self.base_res[1], self.base_res[0])

        # Compute envmap from the MLP
        envmap = self.eval_uv(XY.reshape(-1, 2)).reshape(
            self.base_res[0], self.base_res[1], -1
        )
        # Compute PDF
        Y = XY[..., 1]
        self._pdf = torch.max(envmap, dim=-1)[0] * torch.sin(
            Y * np.pi
        )  # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
        self._pdf[self._pdf <= 0] = 1e-6  # avoid divide by zero in sample()
        self._pdf = self._pdf / torch.sum(self._pdf)  # discrete pdf

        # Compute cumulative sums over the columns and rows
        self.cols = torch.cumsum(self._pdf, dim=1)
        self.rows = torch.cumsum(
            self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0
        )

        # Normalize
        # TODO: for columns/rows with all 0s, use uniform distribution
        self.cols = self.cols / torch.where(
            self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols)
        )
        self.rows = self.rows / torch.where(
            self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows)
        )

        # Prepend 0s to all CDFs
        self.cols = torch.cat([torch.zeros_like(self.cols[:, :1]), self.cols], dim=1)
        self.rows = torch.cat([torch.zeros_like(self.rows[:1, :]), self.rows], dim=0)

    @torch.no_grad()
    def generate_image(self):
        # Evaluate the MLP at the pixel centers
        XY = util.pixel_grid(self.base_res[1], self.base_res[0])

        # Compute envmap from the MLP
        envmap = self.eval_uv(XY.reshape(-1, 2)).reshape(
            self.base_res[0], self.base_res[1], -1
        )

        return envmap.detach().cpu().numpy()
