# from pathlib import Path
import os
import numpy as np
import hydra
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pytorch_lightning as pl

import datasets
from utils.sampler import PatchSampler, UniformSampler, EdgeSampler


def _rgb_to_srgb(f: np.ndarray) -> np.ndarray:
    return np.where(f <= 0.0031308, f * 12.92, np.power(np.clip(f, 0.0031308, None), 1.0/2.4)*1.055 - 0.055)


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


class PeopleSnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, root, subject, split, config, mode, hdri_filepath=None):
        camera = np.load(os.path.join(root, "cameras.npz"))
        K = camera["intrinsic"]
        c2w = np.linalg.inv(camera["extrinsic"])
        height = camera["height"]
        width = camera["width"]

        if mode == "test" and hdri_filepath is not None:
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
        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.npy"))[start:end:skip]

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

        self.split = split
        self.mode = mode
        self.downscale = config.downscale
        self.near = config.get("near", None)
        self.far = config.get("far", None)
        self.image_shape = (height, width)
        if mode == "train":
            self.sampler = hydra.utils.instantiate(config.sampler)

    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_lists[idx]), cv2.COLOR_BGR2RGB)
        msk = np.load(self.msk_lists[idx])
        if self.downscale > 1:
            img = cv2.resize(img, dsize=None, fx=1/self.downscale, fy=1/self.downscale)
            msk = cv2.resize(msk, dsize=None, fx=1/self.downscale, fy=1/self.downscale)

        img = (img[..., :3] / 255).astype(np.float32)
        msk = msk.astype(np.float32)

        if self.mode == "train":
            (msk, img, rays_o, rays_d) = \
                    self.sampler.sample(msk, img, self.rays_o, self.rays_d)
        else:
            rays_o = self.rays_o.reshape(-1, 3)
            rays_d = self.rays_d.reshape(-1, 3)
            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)

        datum = {
            # NeRF
            "rgb": img.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,

            # SMPL parameters
            "betas": self.smpl_params["betas"][0],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],

            # auxiliary
            "alpha": msk,
            # "bg_color": bg_color,
            "index": idx,
            "t_idx": idx / len(self.img_lists),
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
        if self.mode == "test" and hasattr(self, "hdri_filepath"):
            hdri = cv2.cvtColor(
                cv2.imread(
                    self.hdri_filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
                ),
                cv2.COLOR_BGR2RGB,
            )
            # hdri = cv2.resize(hdri, (32, 16), interpolation=cv2.INTER_AREA)
            datum["hdri"] = hdri.astype(np.float32)

        return datum


@datasets.register("peoplesnapshot")
class PeopleSnapshotDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # data_dir = Path(hydra.utils.to_absolute_path(opt.dataroot))
        # for split in ("train", "val", "test"):
        #     dataset = PeopleSnapshotDataset(data_dir, opt.subject, split, opt.get(split))
        #     setattr(self, f"{split}set", dataset)

    def setup(self, stage=None):
        data_root = self.config.dataroot
        if stage in [None, "fit"]:
            split = self.config.train_split
            self.train_dataset = PeopleSnapshotDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="train",
            )
        if stage in [None, "fit", "validate"]:
            split = self.config.val_split
            self.val_dataset = PeopleSnapshotDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="val",
            )
        if stage in [None, "test"]:
            split = self.config.test_split
            hdri_filepath = self.config.get("hdri_filepath", None)
            self.test_dataset = PeopleSnapshotDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="test",
                hdri_filepath=hdri_filepath,
            )
        if stage in [None, "predict"]:
            split = self.config.train_split
            self.predict_dataset = PeopleSnapshotDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="predict",
            )

    def prepare_data(self):
        pass

    # def general_loader(self, dataset, batch_size):
    #     sampler = None
    #     return DataLoader(
    #         dataset,
    #         num_workers=8,
    #         batch_size=batch_size,
    #         pin_memory=True,
    #         sampler=sampler,
    #     )

    # def train_dataloader(self):
    #     return self.general_loader(self.train_dataset, batch_size=1)

    # def val_dataloader(self):
    #     return self.general_loader(self.val_dataset, batch_size=1)

    # def test_dataloader(self):
    #     return self.general_loader(self.test_dataset, batch_size=1)

    # def predict_dataloader(self):
    #     return self.general_loader(self.predict_dataset, batch_size=1)

    def train_dataloader(self):
        if hasattr(self, "train_dataset"):
            return DataLoader(self.train_dataset,
                              shuffle=True,
                              num_workers=self.config.opt.train.num_workers,
                              persistent_workers=True and self.config.opt.train.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            return DataLoader(self.val_dataset,
                              shuffle=False,
                              num_workers=self.config.opt.val.num_workers,
                              persistent_workers=True and self.config.opt.val.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()

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

    def predict_dataloader(self):
        if hasattr(self, "predict_dataset"):
            return DataLoader(self.predict_dataset,
                              shuffle=False,
                              num_workers=self.config.opt.test.num_workers,
                              persistent_workers=True and self.config.opt.test.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().predict_dataloader()
