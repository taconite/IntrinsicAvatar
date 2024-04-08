# from pathlib import Path
import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import datasets


def get_ray_directions(H, W):
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1)
    return xy


def make_rays(K, c2w, H, W):
    xy = get_ray_directions(H, W).reshape(-1, 3).astype(np.float32)
    d_c = xy @ np.linalg.inv(K).T
    d_w = d_c @ c2w[:3, :3].T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    o_w = np.tile(c2w[:3, 3], (len(d_w), 1))
    o_w = o_w.reshape(H, W, 3)
    d_w = d_w.reshape(H, W, 3)
    return o_w.astype(np.float32), d_w.astype(np.float32)


def transform_rays(rays_o, rays_d, c2w):
    """transform rays from camera to world coordinate system"""
    rays_o = rays_o @ c2w[:3, :3].T + c2w[:3, 3]
    rays_d = rays_d @ c2w[:3, :3].T
    return rays_o, rays_d


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }


class AnimationDataset(torch.utils.data.Dataset):
    def __init__(self, root, subject, split, config, betas=None, hdri_filepath=None):
        try:
            cameras = dict(np.load(os.path.join(root, "cameras.npz")))
        except FileNotFoundError:
            # Hack for ZJU-MoCap
            with open(os.path.join(root, "cameras.json"), "r") as f:
                cameras = json.load(f)

            cameras["height"] = 1024
            cameras["width"] = 1024

        # K = camera["intrinsic"]
        # Identity rotation and 0 translation
        c2w = np.eye(4)
        if split == "test" and len(cameras["extrinsic"].shape) == 3:
            height = cameras["height"][0]
            width = cameras["width"][0]
        else:
            height = cameras["height"]
            width = cameras["width"]

        K = np.eye(3)
        K[0, 0] = K[1, 1] = 2000
        K[0, 2] = height // 2
        K[1, 2] = width // 2

        if split == "test" and hdri_filepath is not None:
            self.hdri_filepath = hdri_filepath
            assert os.path.exists(self.hdri_filepath), "HDRI not found: {}".format(
                self.hdri_filepath
            )

        self.downscale = config.downscale
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            K[:2] /= self.downscale

        self.img_wh = (width, height)
        self.has_mask = True

        self.rays_o, self.rays_d = make_rays(K, c2w, height, width)

        # prepare image and mask
        start = config.start
        end = config.end + 1
        skip = config.get("skip", 1)

        if split == "test" and len(cameras["extrinsic"].shape) == 3:
            for k, v in cameras.items():
                cameras[k] = v[start:end:skip]
        self.cameras = cameras

        if split == "train":
            refine = config.get("refine", False)
            if refine: # fix model and optimize SMPL
                cached_path = os.path.join(root, "poses/anim_nerf_test.npz")
            else:
                if os.path.exists(os.path.join(root, f"poses/anim_nerf_{split}.npz")):
                    cached_path = os.path.join(root, f"poses/anim_nerf_{split}.npz")
                elif os.path.exists(os.path.join(root, f"poses/{split}.npz")):
                    cached_path = os.path.join(root, f"poses/{split}.npz")
                else:
                    cached_path = None

            if cached_path and os.path.exists(cached_path):
                print(f"[{split}] Loading from", cached_path)
                self.smpl_params = load_smpl_param(cached_path)
            else:
                print(f"[{split}] No optimized smpl found.")
                self.smpl_params = load_smpl_param(os.path.join(root, "poses.npz"))
                for k, v in self.smpl_params.items():
                    if k != "betas":
                        self.smpl_params[k] = v[start:end:skip]
        elif split == "test":
            smpl_params = dict(np.load(os.path.join(root, "poses.npz")))

            thetas = smpl_params["poses"][..., :72]
            transl = smpl_params["trans"] - smpl_params["trans"][0:1]
            transl += (0, 0.15, 5)

            self.smpl_params = {
                "betas": betas.astype(np.float32).reshape(1, 10),
                "body_pose": thetas[..., 3:].astype(np.float32),
                "global_orient": thetas[..., :3].astype(np.float32),
                "transl": transl.astype(np.float32),
            }
            for k, v in self.smpl_params.items():
                if k != "betas":
                    self.smpl_params[k] = v[start:end:skip]
        else:
            raise ValueError("Invalid split: {}".format(split))

        self.split = split
        self.downscale = config.downscale
        self.near = config.get("near", None)
        self.far = config.get("far", None)
        self.image_shape = (height, width)

    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }

    def __len__(self):
        return len(self.smpl_params["global_orient"])

    def __getitem__(self, idx):
        rays_o = self.rays_o.reshape(-1, 3)
        rays_d = self.rays_d.reshape(-1, 3)

        if self.split == "test":
            if len(self.cameras["extrinsic"].shape) == 3:
                w2c = self.cameras["extrinsic"][idx].astype(np.float32)
            else:
                w2c = self.cameras["extrinsic"].astype(np.float32)
            c2w = np.linalg.inv(w2c)
            rays_o, rays_d = transform_rays(rays_o, rays_d, c2w)

        datum = {
            # NeRF
            "rays_o": rays_o.astype(np.float32),
            "rays_d": rays_d.astype(np.float32),

            # SMPL parameters
            "betas": self.smpl_params["betas"][0],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],

            # "bg_color": bg_color,
            "index": idx,
            "w2c": w2c,
        }
        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            # TODO: we could replace it with bbox in the canonical space
            dist = np.sqrt(np.square(self.smpl_params["transl"][idx]).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)

        if self.split == "test":
            hdri = cv2.cvtColor(
                cv2.imread(
                    self.hdri_filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
                ),
                cv2.COLOR_BGR2RGB,
            )
            # Resize to 2k
            hdri = cv2.resize(hdri, (2048, 1024), interpolation=cv2.INTER_AREA)
            datum["hdri"] = hdri.astype(np.float32)

        return datum


@datasets.register("animation")
class AnimationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        data_root = self.config.dataroot_train
        # Create train dataset to read beta parameters
        split = self.config.train_split
        train_dataset = AnimationDataset(
            data_root, self.config.subject, split, self.config.opt.get(split)
        )
        if stage in [None, "test"]:
            data_root = self.config.dataroot_test
            split = self.config.test_split
            hdri_filepath = self.config.get("hdri_filepath", None)
            self.test_dataset = AnimationDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                betas=train_dataset.smpl_params["betas"],
                hdri_filepath=hdri_filepath,
            )
        else:
            raise ValueError("Invalid stage for AnimationDataModule: {}".format(stage))

    def prepare_data(self):
        pass

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(self.test_dataset,
                              shuffle=False,
                              num_workers=self.config.opt.test.num_workers,
                              persistent_workers=True and self.config.opt.test.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()
