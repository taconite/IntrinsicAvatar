models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(name, config):
    model = models[name](config)
    return model


# IA model
from . import (
    intrinsic_avatar,
)

# Radiance field modules
from .rf import geometry
from .rf import radiance
from .rf import density

# Pose modules
from .pose import pose_encoder
from .pose import pose_correction

# PBR modules
from .pbr import material

# Articulation modules
from .deformers import deformer
from .deformers import non_rigid_deformer

# Import external PBR classes
import lib.torch_pbr

EnvironmentLightTensor = register("envlight-tensor")(lib.torch_pbr.EnvironmentLightTensor)
EnvironmentLightSG = register("envlight-SG")(lib.torch_pbr.EnvironmentLightSG)
EnvironmentLightMLP = register("envlight-mlp")(lib.torch_pbr.EnvironmentLightMLP)
EnvironmentLightNGP = register("envlight-ngp")(lib.torch_pbr.EnvironmentLightNGP)
Mirror = register("brdf-mirror")(lib.torch_pbr.Mirror)
Lambertian = register("brdf-lambertian")(lib.torch_pbr.Lambertian)
GGX = register("brdf-ggx")(lib.torch_pbr.GGX)
DiffuseSGGX = register("phase-diffuse-sggx")(lib.torch_pbr.DiffuseSGGX)
SpecularSGGX = register("phase-specular-sggx")(lib.torch_pbr.SpecularSGGX)
MultiLobe = register("brdf-multi-lobe")(lib.torch_pbr.MultiLobe)
MultiLobeSGGX = register("phase-multi-lobe")(lib.torch_pbr.MultiLobeSGGX)
