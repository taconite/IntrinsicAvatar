import torch
import numpy as np
import torch.nn as nn

import models
from models.base import BaseModel
from models.utils import get_activation, reflect
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


class BaseImplicitRadiance(BaseModel):
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


@models.register('volume-radiance')
class VolumeRadiance(BaseImplicitRadiance):
    def setup(self):
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        xyz_encoding_config = self.config.get('xyz_encoding_config', None)
        xyz_encoding = (
            get_encoding(3, xyz_encoding_config)
            if xyz_encoding_config is not None
            else None
        )
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
        if xyz_encoding is not None:
            self.n_input_dims += xyz_encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.xyz_encoding = xyz_encoding
        self.dir_encoding = dir_encoding
        self.network = network

    def forward(self, points, features, dirs, *args, feature_only=False):
        if self.xyz_encoding is not None:
            points = (points - self.center) / self.scale + 0.5
            xyz_embd = self.xyz_encoding(points.view(-1, 3))
        else:
            xyz_embd = torch.empty(
                features.shape[:1] + (0,), dtype=features.dtype, device=features.device
            )

        network_inp = [xyz_embd, features.view(-1, features.shape[-1])]
        if feature_only:
            return xyz_embd

        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims))
        network_inp.append(dirs_embd)
        network_inp = torch.cat(
            network_inp + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1
        )
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        # Return color and geometric features
        return color, xyz_embd

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        if hasattr(self.network, 'regularizations'):
            return self.network.regularizations()
        else:
            return {}


@models.register('volume-ref-dir-radiance')
class VolumeRefDirRadiance(BaseImplicitRadiance):
    def setup(self):
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        xyz_encoding_config = self.config.get('xyz_encoding_config', None)
        xyz_encoding = (
            get_encoding(3, xyz_encoding_config)
            if xyz_encoding_config is not None
            else None
        )
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
        if xyz_encoding is not None:
            self.n_input_dims += xyz_encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.xyz_encoding = xyz_encoding
        self.dir_encoding = dir_encoding
        self.network = network

        self.register_buffer(
            "sh_mask",
            torch.zeros(1, self.dir_encoding.n_output_dims, dtype=torch.float32),
        )
        # Default parameters for progressive SH bands
        # i.e. no progressive SH
        self.start_step = self.config.get('start_step', 0)
        self.full_band_step = self.config.get('full_band_step', 1)

    def forward(self, points, features, dirs, *args, feature_only=False):
        if self.xyz_encoding is not None:
            points = (points - self.center) / self.scale + 0.5
            xyz_embd = self.xyz_encoding(points.view(-1, 3))
        else:
            xyz_embd = torch.empty(
                features.shape[:1] + (0,), dtype=features.dtype, device=features.device
            )

        network_inp = [xyz_embd, features.view(-1, features.shape[-1])]
        if feature_only:
            return xyz_embd

        dirs = reflect(-dirs, args[0])
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims)) * self.sh_mask
        network_inp.append(dirs_embd)
        network_inp = torch.cat(
            network_inp + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1
        )
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        # Return color and geometric features
        return color, xyz_embd

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

        # Progressively enabling different bands of spherical harmonics
        t = max(global_step - self.start_step, 0.0)
        N = self.full_band_step - self.start_step
        # m = self.dir_encoding.degree
        m = 4
        alpha = m * t / N

        idx = 0
        for deg in range(m):
            w = (
                1.0 - np.cos(np.pi * min(max(alpha - deg, 0.0), 1.0))
            ) / 2.0
            next_idx = idx + deg * 2 + 1
            self.sh_mask[..., idx:next_idx] = w
            idx = next_idx

    def regularizations(self, out):
        if hasattr(self.network, 'regularizations'):
            return self.network.regularizations()
        else:
            return {}


# @models.register('volume-ref-dir-uncertainty-radiance')
# class VolumeRefDirUncertaintyRadiance(BaseImplicitRadiance):
#     def setup(self):
#         self.n_dir_dims = self.config.get('n_dir_dims', 3)
#         self.n_output_dims = 4
#         xyz_encoding_config = self.config.get('xyz_encoding_config', None)
#         xyz_encoding = (
#             get_encoding(3, xyz_encoding_config)
#             if xyz_encoding_config is not None
#             else None
#         )
#         dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
#         self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
#         if xyz_encoding is not None:
#             self.n_input_dims += xyz_encoding.n_output_dims
#         network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
#         self.xyz_encoding = xyz_encoding
#         self.dir_encoding = dir_encoding
#         self.network = network
#
#         self.register_buffer(
#             "sh_mask",
#             torch.zeros(1, self.dir_encoding.n_output_dims, dtype=torch.float32),
#         )
#         self.start_step = self.config.get('start_step', 0)
#         self.full_band_step = self.config.get('full_band_step', 1)
#
#     def forward(self, points, features, dirs, *args, feature_only=False):
#         if self.xyz_encoding is not None:
#             points = (points - self.center) / self.scale + 0.5
#             xyz_embd = self.xyz_encoding(points.view(-1, 3))
#         else:
#             xyz_embd = torch.empty(
#                 features.shape[:1] + (0,), dtype=features.dtype, device=features.device
#             )
#
#         network_inp = [xyz_embd, features.view(-1, features.shape[-1])]
#         if feature_only:
#             return torch.cat(network_inp, dim=-1)
#
#         dirs = reflect(-dirs, args[0])
#         dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
#         dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims)) * self.sh_mask
#         network_inp.append(dirs_embd)
#         network_inp = torch.cat(
#             network_inp + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1
#         )
#         out = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
#         color = out[..., :3]
#         uncertainty = torch.nn.functional.softplus(out[..., 3:])
#         if 'color_activation' in self.config:
#             color = get_activation(self.config.color_activation)(color)
#         # Return color and geometric features
#         out = torch.cat([color, uncertainty], dim=-1)
#         return out, torch.cat(
#             [xyz_embd, features.view(-1, features.shape[-1])], dim=-1
#         )
#
#     def update_step(self, epoch, global_step):
#         update_module_step(self.dir_encoding, epoch, global_step)
#         update_module_step(self.xyz_encoding, epoch, global_step)
#
#         # Progressively enabling different bands of spherical harmonics
#         t = max(global_step - self.start_step, 0.0)
#         N = self.full_band_step - self.start_step
#         # m = self.dir_encoding.degree
#         m = 4
#         alpha = m * t / N
#
#         idx = 0
#         for deg in range(m):
#             w = (
#                 1.0 - np.cos(np.pi * min(max(alpha - deg, 0.0), 1.0))
#             ) / 2.0
#             next_idx = idx + deg * 2 + 1
#             self.sh_mask[..., idx:next_idx] = w
#             idx = next_idx
#
#     def regularizations(self, out):
#         if hasattr(self.network, 'regularizations'):
#             return self.network.regularizations()
#         else:
#             return {}


# IDE-based radiance field from RefNeRF
@models.register('volume-reflection-radiance')
class VolumeReflectionRadiance(BaseImplicitRadiance):
    def setup(self):
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        xyz_encoding_config = self.config.get('xyz_encoding_config', None)
        xyz_encoding = (
            get_encoding(3, xyz_encoding_config)
            if xyz_encoding_config is not None
            else None
        )
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim
        if xyz_encoding is not None:
            self.n_input_dims += xyz_encoding.n_output_dims
        self.n_bottleneck_dims = self.config.get('n_bottleneck_dims', self.n_input_dims)
        self.roughness_bias = self.config.get('roughness_bias', 1.0)
        network = get_mlp(
            self.n_input_dims + dir_encoding.n_output_dims + 1,
            self.n_output_dims,
            self.config.mlp_network_config,
        )
        self.xyz_encoding = xyz_encoding
        self.dir_encoding = dir_encoding
        self.network = network

        self.roughness_layer = nn.Linear(self.n_input_dims, 1)
        self.diffuse_layer = nn.Linear(self.n_input_dims, 3)
        self.tint_layer = nn.Linear(self.n_input_dims, 3)
        self.bottleneck_layer = nn.Linear(self.n_input_dims, self.n_bottleneck_dims)

        # Activation for roughness
        self.softplus = nn.Softplus()
        self.register_buffer(
            "sh_mask",
            torch.zeros(1, self.dir_encoding.n_output_dims, dtype=torch.float32),
        )
        self.start_step = self.config.get('start_step', 0)
        self.full_band_step = self.config.get('full_band_step', 1)

    def forward(self, points, features, dirs, normals, *args, feature_only=False):
        if self.xyz_encoding is not None:
            points = (points - self.center) / self.scale + 0.5
            network_inp = torch.cat(
                [self.xyz_encoding(points.view(-1, 3)), features]
                + [arg.view(-1, arg.shape[-1]) for arg in args],
                dim=-1,
            )
        else:
            network_inp = torch.cat(
                [features.view(-1, features.shape[-1])]
                + [arg.view(-1, arg.shape[-1]) for arg in args],
                dim=-1,
            )

        geometric_features = network_inp
        if feature_only:
            return torch.cat(network_inp, dim=-1)
        # Dot product between normal vectors and view directions.
        dotprod = torch.sum(-dirs * normals, dim=-1, keepdim=True)

        # Predict diffuse color
        raw_rgb_diffuse = self.diffuse_layer(network_inp)

        # Predict specular tint
        tint = torch.sigmoid(self.tint_layer(network_inp))

        # Predict roughness
        roughness = self.softplus(
            self.roughness_layer(network_inp) + self.roughness_bias
        )
        bottleneck = self.bottleneck_layer(network_inp)
        if self.training:
            bottleneck += torch.normal(
                mean=0,
                std=1,
                size=bottleneck.shape,
                dtype=bottleneck.dtype,
                device=bottleneck.device,
            )

        # Reflect viewing directions about normal
        ref_dirs = reflect(-dirs, normals)
        # Encode reflected viewing directions
        if self.config.dir_encoding_config.otype == 'IDE':
            dirs_embd = self.dir_encoding(ref_dirs, roughness)
        elif self.config.dir_encoding_config.otype == 'SphericalHarmonics':
            ref_dirs = (ref_dirs + 1.) / 2.  # (-1, 1) => (0, 1)
            dirs_embd = self.dir_encoding(ref_dirs.view(-1, self.n_dir_dims))
        else:
            raise ValueError('Unknown dir encoding type', self.config.dir_encoding_config.otype)

        network_inp = torch.cat(
            [
                bottleneck,
                dirs_embd * self.sh_mask,
                dotprod,
            ],
            dim=-1,
        )

        specular = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            specular = get_activation(self.config.color_activation)(specular)

        diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
        specular_linear = (tint * specular)

        return specular_linear + diffuse_linear, geometric_features

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)
        # Progressively enabling different bands of spherical harmonics
        t = max(global_step - self.start_step, 0.0)
        N = self.full_band_step - self.start_step
        # m = self.dir_encoding.degree
        m = 4
        alpha = m * t / N

        idx = 0
        for deg in range(m):
            w = (
                1.0 - np.cos(np.pi * min(max(alpha - deg, 0.0), 1.0))
            ) / 2.0
            next_idx = idx + deg * 2 + 1
            self.sh_mask[..., idx:next_idx] = w
            idx = next_idx

    def regularizations(self, out):
        if hasattr(self.network, 'regularizations'):
            return self.network.regularizations()
        else:
            return {}
