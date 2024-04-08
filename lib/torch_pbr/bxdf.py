import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import models
from .utils.warp_utils import (
    coordinate_system,
    to_local,
    to_world,
    # gen_stratified_samples,
    fresnel_schlick,
    # smith_GGX_G1_aniso,
    smith_GGX_G1_shclick,
    sample_Lambertian_surface,
    sample_uniform_hemisphere,
    sample_GGX_VNDF,
    sample_specular_SGGX,
    # sample_diffuse_SGGX,
    eval_Lambertian_surface,
    eval_uniform_hemisphere,
    eval_GGX_NDF,
    eval_GGX_VNDF,
    eval_diffuse_SGGX,
    eval_specular_SGGX,
)
from .utils import nvdiffrecmc_util as util


class BaseScatterer(nn.Module):
    def __init__(self, config):
        super(BaseScatterer, self).__init__()
        self.config = config

    def pdf(self, wi, n, wo, **kwargs):
        raise NotImplementedError

    def eval(self, wi, n, wo, **kwargs):
        raise NotImplementedError

    def eval_with_cosine(self, wi, n, wo, **kwargs):
        raise NotImplementedError

    def sample(self, n, wi, **kwargs):
        raise NotImplementedError

    def perfect_sampling(self):
        raise NotImplementedError

    def is_delta(self):
        raise NotImplementedError

    def surface_scattering(self):
        raise NotImplementedError


class BaseBxDF(BaseScatterer):
    def __init__(self, config):
        super(BaseBxDF, self).__init__(config)

    def surface_scattering(self):
        return True


class BasePhaseFunction(BaseScatterer):
    def __init__(self, config):
        super(BasePhaseFunction, self).__init__(config)

    def surface_scattering(self):
        return False


# @models.register("brdf-mirror")
class Mirror(BaseBxDF):
    def __init__(self, config):
        super(Mirror, self).__init__(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        return torch.zeros_like(wi[..., :1])

    @torch.no_grad()
    def eval(self, wi, n, wo, **kwargs):
        spec = self.eval_with_cosine(wi, n, wo, **kwargs)
        diff = torch.zeros_like(spec)
        return diff, spec.repeat(1, 3)

    @torch.no_grad()
    def eval_with_cosine(self, wi, n, wo, **kwargs):
        return torch.where((wi * n).sum(-1, keepdim=True) <= 0, torch.zeros_like(wi[..., :1]), torch.ones_like(wi[..., :1]))

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        # t, b = coordinate_system(n)
        # wi = to_local(wi, t, b, n)
        # wo = torch.stack([-wi[..., 0], -wi[..., 1], wi[..., 2]], dim=-1)
        # wo = to_world(wo, t, b, n)
        # return wo
        # Reflect wi abount n
        wo = 2 * (wi * n).sum(-1, keepdim=True) * n - wi
        return wo

    def perfect_sampling(self):
        return False

    def is_delta(self):
        return True


# @models.register("brdf-lambertian")
class Lambertian(BaseBxDF):
    def __init__(self, config):
        super(Lambertian, self).__init__(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        if self.training:
            pdf = eval_uniform_hemisphere(wo, n)
        else:
            pdf = eval_Lambertian_surface(wo, n)
        return pdf.unsqueeze(-1)

    def eval(self, wi, n, wo, **kwargs):
        diff = self.eval_with_cosine(wi, n, wo, **kwargs)
        spec = torch.zeros(len(diff), 3, device=diff.device, dtype=diff.dtype)
        return diff, spec

    def eval_with_cosine(self, wi, n, wo, **kwargs):
        return F.relu((wo * n).sum(-1, keepdim=True)) / np.pi

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        if self.training:
            wo = sample_uniform_hemisphere(sample, n)
        else:
            wo = sample_Lambertian_surface(sample, n)
        return wo

    def perfect_sampling(self):
        return False    # for inverse rendering cosine-weighted hemisphere sampling is not perfect

    def is_delta(self):
        return False


# Note: there is no closed-form solution for pdf() of diffuse SGGX. We use the same pdf() as
# Lambertian for now.
# @models.register("phase-diffuse-sggx")
class DiffuseSGGX(BasePhaseFunction):
    def __init__(self, config):
        super(DiffuseSGGX, self).__init__(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        pdf = eval_uniform_hemisphere(wo, n)
        return pdf.unsqueeze(-1)

    def eval(self, wi, n, wo, **kwargs):
        diff = self.eval_with_cosine(wi, n, wo, **kwargs)
        spec = torch.zeros(len(diff), 3, device=diff.device, dtype=diff.dtype)
        return diff, spec

    # Phase function does not need the cosine term - we just use the method name for compatibility
    # with the Scatterer interface
    def eval_with_cosine(self, wi, n, wo, **kwargs):
        assert (kwargs["alpha_x"] == kwargs["alpha_y"]).all()
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        return eval_diffuse_SGGX(sample, wi, n, wo, kwargs["alpha_x"]).unsqueeze(-1)

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        wo = sample_uniform_hemisphere(sample, n)
        return wo

    def perfect_sampling(self):
        return True

    def is_delta(self):
        return False


# @models.register("phase-specular-sggx")
class SpecularSGGX(BasePhaseFunction):
    def __init__(self, config):
        super(SpecularSGGX, self).__init__(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        pdf = eval_specular_SGGX(wi, n, wo, kwargs["alpha_x"])
        return pdf.unsqueeze(-1)

    def eval(self, wi, n, wo, **kwargs):
        spec = self.eval_with_cosine(wi, n, wo, **kwargs)
        diff = torch.zeros_like(spec)
        return diff, spec.repeat(1, 3)

    def eval_with_cosine(self, wi, n, wo, **kwargs):
        assert (kwargs["alpha_x"] == kwargs["alpha_y"]).all()
        return eval_specular_SGGX(wi, n, wo, kwargs["alpha_x"]).unsqueeze(-1)

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        wo = sample_specular_SGGX(sample, n, wi, kwargs["alpha_x"])
        return wo

    def perfect_sampling(self):
        return True

    def is_delta(self):
        return False


# @models.register("brdf-ggx")
class GGX(BaseBxDF):
    def __init__(self, config):
        super(GGX, self).__init__(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        eps = 1e-6
        t, b = coordinate_system(n)
        wo = to_local(wo, t, b, n)
        wi = to_local(wi, t, b, n)
        wh = wi + wo
        wh = F.normalize(wh, dim=-1)
        pdf = torch.where(
            4 * (wi * wh).sum(-1).abs() > eps,
            eval_GGX_VNDF(wh, wi, alpha_x=kwargs["alpha_x"], alpha_y=kwargs["alpha_y"])
            / (4 * (wo * wh).sum(-1).abs() + eps),
            torch.zeros_like(wo[..., 0]),
        )

        return pdf.unsqueeze(-1)

    def eval(self, wi, n, wo, **kwargs):
        spec = self.eval_with_cosine(wi, n, wo, **kwargs)
        diff = torch.zeros(len(spec), 1, device=spec.device, dtype=spec.dtype)
        return diff, spec

    def eval_with_cosine(self, wi, n, wo, **kwargs):
        t, b = coordinate_system(n)
        wo = to_local(wo, t, b, n)
        wi = to_local(wi, t, b, n)
        wh = wi + wo
        wh = F.normalize(wh, dim=-1)

        eps = 1e-6
        F0 = kwargs.get("F0", 0.04 * torch.ones_like(wi))
        alpha = kwargs["alpha_x"]
        # TODO: check the deriviation of k
        k = (alpha ** 2 + 2 * alpha + 1) / 8.0
        return torch.where(
            torch.logical_and(wi[:, 2:] > eps, wo[:, 2:] > eps),
            eval_GGX_NDF(
                wh, alpha_x=kwargs["alpha_x"], alpha_y=kwargs["alpha_y"]
            ).unsqueeze(-1)
            * smith_GGX_G1_shclick(wi, k).unsqueeze(-1)
            * smith_GGX_G1_shclick(wo, k).unsqueeze(-1)
            * fresnel_schlick(F0, 1.0, (wi * wh).sum(-1, keepdim=True).abs())
            / (4 * wi[:, 2:] + eps),
            torch.zeros_like(wi),
        )

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        t, b = coordinate_system(n)
        wi = to_local(wi, t, b, n)
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        wh = sample_GGX_VNDF(
            sample,
            wi,
            alpha_x=kwargs["alpha_x"],
            alpha_y=kwargs["alpha_y"],
        )
        wo = 2 * (wi * wh).sum(dim=-1, keepdim=True) * wh - wi
        wo = to_world(wo, t, b, n)
        return wo

    def perfect_sampling(self):
        return False

    def is_delta(self):
        return False


# @models.register("brdf-multi-lobe")
class MultiLobe(BaseBxDF):
    def __init__(self, config):
        super(MultiLobe, self).__init__(config)
        self.diffuse = Lambertian(config)
        self.specular = GGX(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        # if self.training:
        #     return eval_uniform_hemisphere(wo, n).unsqueeze(-1)
        # else:
        metallic = kwargs["metallic"]
        kd = kwargs["albedo"]
        weight_diffuse = (1.0 - metallic) * util.luminance(kd)
        cos_theta = (wi * n).sum(-1, keepdim=True)
        weight_specular = torch.where(
            cos_theta > 0,
            util.luminance(fresnel_schlick(kd, 1.0, cos_theta)),
            torch.zeros_like(cos_theta),
        )
        eps = 1e-6
        p_diffuse = torch.where(
            weight_diffuse + weight_specular > eps,
            weight_diffuse / (weight_diffuse + weight_specular + eps),
            torch.ones_like(weight_diffuse),
        )
        p_specular = 1 - p_diffuse
        return p_diffuse * self.diffuse.pdf(
            wi, n, wo, **kwargs
        ) + p_specular * self.specular.pdf(wi, n, wo, **kwargs)

    def eval(self, wi, n, wo, **kwargs):
        metallic = kwargs["metallic"]
        attenuation = kwargs["attenuation"]
        kd = kwargs["albedo"]
        F0 = (0.04 * (1.0 - metallic) + kd * metallic) * (1.0 - attenuation)
        # kd = (1.0 - metallic) * kd
        diff = self.diffuse.eval_with_cosine(wi, n, wo, **kwargs)
        spec = self.specular.eval_with_cosine(wi, n, wo, F0=F0, **kwargs)

        return diff, spec

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        # if self.training:
        #     sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        #     batch_size = n.shape[0] // 256
        #     sample = gen_stratified_samples(batch_size, 16, 16, n.device, self.training)
        #     return sample_uniform_hemisphere(sample, n)
        # else:
        # Compute probability of sampling the specular component
        metallic = kwargs["metallic"]
        kd = kwargs["albedo"]
        weight_diffuse = (1.0 - metallic) * util.luminance(kd)
        cos_theta = (wi * n).sum(-1, keepdim=True)
        weight_specular = torch.where(
            cos_theta > 0,
            util.luminance(fresnel_schlick(kd, 1.0, cos_theta)),
            torch.zeros_like(cos_theta),
        )
        eps = 1e-6
        p_specular = torch.where(
            weight_diffuse + weight_specular > eps,
            weight_specular / (weight_diffuse + weight_specular + eps),
            torch.zeros_like(weight_diffuse),
        ).squeeze(-1)
        # Get random variables
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        kwargs.pop("sample", None)
        # Use the first random variable to select between specular and diffuse
        specular_mask = p_specular > sample[:, 0]
        specular_sample = sample[specular_mask].clone()
        diffuse_sample = sample[~specular_mask].clone()
        # Rescale the first random variable to [0, 1]
        specular_sample[:, 0] = specular_sample[:, 0] / p_specular[specular_mask]
        diffuse_sample[:, 0] = (diffuse_sample[:, 0] - p_specular[~specular_mask]) / (
            1 - p_specular[~specular_mask]
        )
        # Sample diffuse and specular lobes
        wo = torch.tensor([[0, 0, 1]], dtype=wi.dtype, device=wi.device).repeat(
            wi.shape[0], 1
        )
        if specular_mask.sum() > 0:
            kwargs_specular = {k: v[specular_mask] for k, v in kwargs.items()}
            kwargs_specular["sample"] = specular_sample
            wo_specular = self.specular.sample(
                n[specular_mask], wi[specular_mask], **kwargs_specular
            )
            wo.masked_scatter_(specular_mask.unsqueeze(-1), wo_specular)

        if (~specular_mask).sum() > 0:
            kwargs_diffuse = {k: v[~specular_mask] for k, v in kwargs.items()}
            kwargs_diffuse["sample"] = diffuse_sample
            wo_diffuse = self.diffuse.sample(
                n[~specular_mask], wi[~specular_mask], **kwargs_diffuse
            )
            wo.masked_scatter_(~specular_mask.unsqueeze(-1), wo_diffuse)

        return wo

    def perfect_sampling(self):
        return False

    def is_delta(self):
        return False


#TODO: find a way to blend specular and diffuse SGGX...
# @models.register("phase-multi-lobe")
class MultiLobeSGGX(BaseBxDF):
    def __init__(self, config):
        super(MultiLobeSGGX, self).__init__(config)
        self.diffuse = DiffuseSGGX(config)
        self.specular = SpecularSGGX(config)

    @torch.no_grad()
    def pdf(self, wi, n, wo, **kwargs):
        ks = kwargs["metallic"]
        kd = kwargs["albedo"]
        weight_diffuse = util.luminance(kd)
        weight_specular = util.luminance(ks)
        eps = 1e-6
        p_diffuse = torch.where(
            weight_diffuse + weight_specular > eps,
            weight_diffuse / (weight_diffuse + weight_specular + eps),
            torch.ones_like(weight_diffuse),
        )
        p_specular = 1 - p_diffuse
        return p_diffuse * self.diffuse.pdf(
            wi, n, wo, **kwargs
        ) + p_specular * self.specular.pdf(wi, n, wo, **kwargs)

    def eval(self, wi, n, wo, **kwargs):
        diff = self.diffuse.eval_with_cosine(wi, n, wo, **kwargs)
        spec = self.specular.eval_with_cosine(wi, n, wo, **kwargs)

        return diff, spec

    @torch.no_grad()
    def sample(self, n, wi, **kwargs):
        # Compute probability of sampling the specular component
        ks = kwargs["metallic"]
        kd = kwargs["albedo"]
        weight_diffuse = util.luminance(kd)
        weight_specular = util.luminance(ks)
        eps = 1e-6
        p_specular = torch.where(
            weight_diffuse + weight_specular > eps,
            weight_specular / (weight_diffuse + weight_specular + eps),
            torch.zeros_like(weight_diffuse),
        ).squeeze(-1)
        # Get random variables
        sample = kwargs.get("sample", torch.rand(wi.shape[0], 2, device=wi.device))
        kwargs.pop("sample", None)
        # Use the first random variable to select between specular and diffuse
        specular_mask = p_specular > sample[:, 0]
        specular_sample = sample[specular_mask].clone()
        diffuse_sample = sample[~specular_mask].clone()
        # Rescale the first random variable to [0, 1]
        specular_sample[:, 0] = specular_sample[:, 0] / p_specular[specular_mask]
        diffuse_sample[:, 0] = (diffuse_sample[:, 0] - p_specular[~specular_mask]) / (
            1 - p_specular[~specular_mask]
        )
        # Sample diffuse and specular lobes
        wo = torch.tensor([[0, 0, 1]], dtype=wi.dtype, device=wi.device).repeat(
            wi.shape[0], 1
        )
        if specular_mask.sum() > 0:
            kwargs_specular = {k: v[specular_mask] for k, v in kwargs.items()}
            kwargs_specular["sample"] = specular_sample
            wo_specular = self.specular.sample(
                n[specular_mask], wi[specular_mask], **kwargs_specular
            )
            wo.masked_scatter_(specular_mask.unsqueeze(-1), wo_specular)

        if (~specular_mask).sum() > 0:
            kwargs_diffuse = {k: v[~specular_mask] for k, v in kwargs.items()}
            kwargs_diffuse["sample"] = diffuse_sample
            wo_diffuse = self.diffuse.sample(
                n[~specular_mask], wi[~specular_mask], **kwargs_diffuse
            )
            wo.masked_scatter_(~specular_mask.unsqueeze(-1), wo_diffuse)

        return wo

    def perfect_sampling(self):
        return False

    def is_delta(self):
        return False
