import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.utils import get_activation
from models.network_utils import get_mlp
from models.utils import GaussianHistogram
from lib.torch_pbr import luminance
# from systems.utils import update_module_step


@models.register("volume-material")
class VolumeMaterial(nn.Module):
    def __init__(self, config):
        super(VolumeMaterial, self).__init__()
        self.config = config
        self.n_output_dims = self.config.get("n_output_dim", 5)
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(
            self.n_input_dims, self.n_output_dims, self.config.mlp_network_config
        )
        self.network = network
        self.albedo_bias = self.config.get("albedo_bias", 0.03)
        self.albedo_scale = self.config.get("albedo_scale", 0.77)
        self.roughness_bias = self.config.get("roughness_bias", 0.09)
        self.roughness_scale = self.config.get("roughness_scale", 0.9)
        self.metallic_bias = self.config.get("metallic_bias", 0.0)
        self.metallic_scale = self.config.get("metallic_scale", 1.0)

    def forward(self, features, *args):
        network_inp = torch.cat(
            [features.view(-1, features.shape[-1])]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        material = (
            self.network(network_inp)
            .view(*features.shape[:-1], self.n_output_dims)
            .float()
        )
        # material[..., 3:4] = material[..., 3:4] + self.roughness_bias
        # material[..., 4:] = material[..., 4:] + self.metallic_bias
        if "material_activation" in self.config:
            material = get_activation(self.config.material_activation)(material)
        albedo = material[..., :3] * self.albedo_scale + self.albedo_bias
        roughness = (
            material[..., 3:4] * self.roughness_scale + self.roughness_bias
        )
        metallic = material[..., 4:] * self.metallic_scale + self.metallic_bias
        return torch.cat([albedo, roughness, metallic], dim=-1)

    def regularizations(self, out):
        normal_orientation = out["normals_orientation_loss_map"].mean()
        albedo_smoothness = out["albedo_smoothness_loss_map"].mean()
        roughness_smoothness = out["roughness_smoothness_loss_map"].mean()
        metallic_smoothness = out["metallic_smoothness_loss_map"].mean()
        albedo_pred = torch.log(out["comp_albedo_full"][out["rays_valid_phys_full"][..., 0]] + 1e-6)
        albedo_entropy = 0
        for i in range(albedo_pred.shape[-1]):
            channel = albedo_pred[..., i]
            hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
            h = hist(channel)
            if h.sum() > 1e-6:
                h = h.div(h.sum()) + 1e-6
            else:
                h = torch.ones_like(h).to(h)
            albedo_entropy += torch.sum(-h*torch.log(h))

        ret = {
            "normal_orientation": normal_orientation,
            "albedo_smoothness": albedo_smoothness,
            "roughness_smoothness": roughness_smoothness,
            "metallic_smoothness": metallic_smoothness,
            "albedo_entropy": albedo_entropy,
        }

        # Energy conservation loss if specular albedo is predicted by the network
        if out["comp_metallic_full"].size(-1) == 3:
            diffuse_albedo = out["comp_albedo_full"][out["rays_valid_phys_full"][..., 0]]
            specular_albedo = out["comp_metallic_full"][out["rays_valid_phys_full"][..., 0]]
            luminance_loss = F.relu(
                luminance(diffuse_albedo) + luminance(specular_albedo) - 1.0
            )
            ret.update({"energy_conservation": luminance_loss.mean()})

        return ret
