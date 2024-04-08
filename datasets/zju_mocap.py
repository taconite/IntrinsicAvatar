# from pathlib import Path
import os
import json
import numpy as np
import hydra
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pytorch_lightning as pl

import datasets

from .utils import KeyIndex


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


# Check if rays defined by (rays_o, rays_d) intersect with the given aabb
def intersect_aabb(aabb, rays_o, rays_d):
    tmin = (aabb[0] - rays_o) / rays_d
    tmax = (aabb[1] - rays_o) / rays_d
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    tmin = np.max(t1, axis=-1)
    tmax = np.min(t2, axis=-1)
    return np.logical_and(tmax >= 0, tmin <= tmax)


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


class ZJUMoCapDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, subject, split, config, mode, hdri_filepath=None):
        root = os.path.join(data_root, subject)
        with open(os.path.join(root, "cameras.json"), "r") as f:
            cameras = json.load(f)

        # self.all_cam_names = cameras["all_cam_names"]
        if subject in ["CoreView_313", "CoreView_315"]:
            cam_names = cameras["all_cam_names"]
        else:
            cam_names = ["Camera_B1"]

		# For multi-view dataset, We assume:
		# 1) image resolutions are the same across all cameras
		# 2) frame indices are the same across all cameras
        # 3) image name can be properly sorted by the sorted() function
        start = config.start
        end = config.end + 1
        skip = config.get("skip", 1)
        frame_indices = list(range(start, end, skip))
        data_indices = list(range(len(frame_indices)))
        total_frames = [len(frame_indices)] * len(frame_indices)
        self.index = (
            KeyIndex(cam_names, "camera")
            * (
                KeyIndex(frame_indices, "frame")
                + KeyIndex(data_indices, "data_idx")
                + KeyIndex(total_frames, "total_frames")
            )
        ).to_list()

        self.kernel = config.get("kernel", 5)

        self.downscale = config.downscale

        self.K = {}
        self.dist = {}
        self.w2c = {}
        self.camera_loc = {}
        self.rays_o = {}
        self.rays_d = {}
        # Compute camera specifc information:
        # 1) camera intrinsics (K, dist)
        # 2) camera extrinsics (w2c)
        # 3) camera location (camera_loc)
        # 4) camera rays (rays_o, rays_d)
        # 5) image list (img_lists)
        # 6) boundary mask list (bounds_lists, specific for the ZJU-MoCap dataset)
        # 7) mask list (msk_lists)
        self.img_lists = []
        self.bounds_lists = []
        self.msk_lists = []
        for cam_idx, cam_name in enumerate(cam_names):
            camera = cameras[cam_name]
            K = np.array(camera["intrinsic"], dtype=np.float32)
            RT = np.array(camera["extrinsic"], dtype=np.float32)
            dist = np.array(camera["distortion"], dtype=np.float32)
            c2w = np.linalg.inv(RT)
            height = camera["height"]
            width = camera["width"]

            self.K[cam_name] = K.copy()
            self.dist[cam_name] = dist.copy()
            self.w2c[cam_name] = RT.copy()
            self.camera_loc[cam_name] = self.get_camera_location(RT[:3, :3], RT[:3, 3])

            if self.downscale > 1:
                height = int(height / self.downscale)
                width = int(width / self.downscale)
                K[:2] /= self.downscale

            # We assume image resolutions are the same across all cameras
            if cam_idx == 0:
                self.img_wh = (width, height)
                self.has_mask = True

            self.rays_o[cam_name], self.rays_d[cam_name] = make_rays(K, c2w, height, width)

            self.img_lists.extend(
                sorted(glob.glob(f"{root}/images/{cam_name}/*.jpg"))[start:end:skip]
            )
            self.bounds_lists.extend(
                (glob.glob(f"{root}/bound_masks/{cam_name}/*.png"))[start:end:skip]
            )
            # `refined_*`=SAM masks, `mask_*`=original masks
            self.msk_lists.extend(
                sorted(glob.glob(f"{root}/masks/{cam_name}/mask_*.png"))[
                    start:end:skip
                ]
            )

        # SMPL parameters
        self.smpl_params = load_smpl_param(os.path.join(root, "poses.npz"))
        for k, v in self.smpl_params.items():
            if k != "betas":
                self.smpl_params[k] = v[start:end:skip]

        self.mode = mode
        self.split = split
        self.near = config.get("near", None)
        self.far = config.get("far", None)
        self.image_shape = (height, width)
        if mode == "train":
            self.sampler = hydra.utils.instantiate(config.sampler)

        if mode == "test" and hdri_filepath is not None:
            self.hdri_filepath = hdri_filepath
            assert os.path.exists(self.hdri_filepath), "HDRI not found: {}".format(
                self.hdri_filepath
            )

    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }

    def get_camera_location(self, R, T):
        # Note: R is not orthogonal for othographic camera!
        return R.T @ -T

    def __len__(self):
        return len(self.img_lists)

    def get_mask(self, mask_in, kernel=5):
        mask = (mask_in != 0).astype(np.uint8)

        kernel = np.ones((kernel, kernel), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        mask[(mask_dilate - mask_erode) == 1] = 100

        return mask

    def __getitem__(self, idx):
        index = self.index[idx]
        cam_name = index["camera"]
        data_idx = index["data_idx"]
        # frame_idx = index["frame_idx"]
        total_frames = index["total_frames"]

        img = cv2.cvtColor(cv2.imread(self.img_lists[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msk_lists[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.undistort(img, self.K[cam_name], self.dist[cam_name], None)
        msk = cv2.undistort(msk, self.K[cam_name], self.dist[cam_name], None)
        msk = (msk > 0).astype(np.uint8)

        bound_msk = cv2.imread(self.bounds_lists[idx], cv2.IMREAD_GRAYSCALE)
        msk_erode = self.get_mask(msk, self.kernel) if self.kernel > 0 else msk.copy()
        if self.downscale > 1:
            img = cv2.resize(
                img,
                dsize=None,
                fx=1 / self.downscale,
                fy=1 / self.downscale,
                interpolation=cv2.INTER_AREA,
            )
            bound_msk = cv2.resize(
                bound_msk,
                dsize=None,
                fx=1 / self.downscale,
                fy=1 / self.downscale,
                interpolation=cv2.INTER_NEAREST,
            )
            msk = cv2.resize(
                msk,
                dsize=None,
                fx=1 / self.downscale,
                fy=1 / self.downscale,
                interpolation=cv2.INTER_NEAREST,
            )
            msk_erode = cv2.resize(
                msk_erode,
                dsize=None,
                fx=1 / self.downscale,
                fy=1 / self.downscale,
                interpolation=cv2.INTER_NEAREST,
            )

        msk_combined = 100 * np.ones(msk.shape, dtype=np.uint8)
        msk_combined[msk_erode == 1] = 1
        msk_combined[(msk_erode == 0) & (bound_msk == 1)] = 0
        img = (img[..., :3] / 255).astype(np.float32)

        # Also compute foreground region mask for evaluation purpose
        # we dilate the foreground region mask to account for potential
        # pose misalignment for novel pose relighting evaluation
        # NOTE: Just for compatibility with the system/model, not really used
        # for this dataset
        dilate_kernel = np.ones((50, 50), np.uint8)
        msk_dilate = cv2.dilate(msk.astype(np.uint8), dilate_kernel, iterations=1)
        x, y, w, h = cv2.boundingRect(msk_dilate)
        valid_msk = np.zeros(msk_dilate.shape, dtype=bool)
        valid_msk[y : y + h, x : x + w] = True

        if self.mode == "train":
            (msk, img, valid_msk, rays_o, rays_d) = self.sampler.sample(
                msk_combined,
                img,
                valid_msk,
                self.rays_o[cam_name],
                self.rays_d[cam_name],
            )
        else:
            rays_o = self.rays_o[cam_name].reshape(-1, 3)
            rays_d = self.rays_d[cam_name].reshape(-1, 3)
            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)
            msk_erode = msk_erode.reshape(-1)
            valid_msk = valid_msk.reshape(-1)

        datum = {
            # NeRF
            "rgb": img.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,

            # SMPL parameters
            "betas": self.smpl_params["betas"][0],
            "global_orient": self.smpl_params["global_orient"][data_idx],
            "body_pose": self.smpl_params["body_pose"][data_idx],
            "transl": self.smpl_params["transl"][data_idx],

            # auxiliary
            "alpha": msk,
            "valid_mask": valid_msk,
            # "bg_color": bg_color,
            "index": data_idx,
            "t_idx": data_idx / total_frames,
            "w2c": self.w2c[cam_name].astype(np.float32),
        }
        # near and far are not used in the model
        # instead, we use a fixed bounding box as we ray-marching is onde in a pelvis-aligned
        # coordinate system
        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            dist = np.sqrt(
                np.square(
                    self.camera_loc[cam_name] - self.smpl_params["transl"][data_idx]
                ).sum(-1)
            )
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


@datasets.register("zju-mocap")
class ZJUMoCapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        data_root = self.config.dataroot
        if stage in [None, "fit"]:
            split = self.config.train_split
            self.train_dataset = ZJUMoCapDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="train",
            )
        if stage in [None, "fit", "validate"]:
            split = self.config.val_split
            self.val_dataset = ZJUMoCapDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="val",
            )
        if stage in [None, "test"]:
            split = self.config.test_split
            hdri_filepath = self.config.get("hdri_filepath", None)
            self.test_dataset = ZJUMoCapDataset(
                data_root,
                self.config.subject,
                split,
                self.config.opt.get(split),
                mode="test",
                hdri_filepath=hdri_filepath,
            )
        if stage in [None, "predict"]:
            split = self.config.train_split
            self.predict_dataset = ZJUMoCapDataset(
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
