import wandb
import torch
import lpips
import numpy as np
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
from torch.utils.data import DataLoader

# from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

from models.utils import compute_albedo_rescale_factor
from models.occ_grid.temporal_occ_grid import TemporalOccGridEstimator
from lib.torch_pbr import rgb_to_srgb, luma, max_value
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, NormalError, SSIM, LPIPS, binary_cross_entropy


@systems.register('intrinsic-avatar-system')
class IntrinsicAvatarSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
            'lpips': LPIPS(),
            'normal_error': NormalError()
        }
        self.reinit_optimizer_steps = self.config.system.get(
            "reinit_optimizer_steps", [-1]
        )
        self.reinit_occupancy_grid_steps = self.config.system.get(
            "reinit_occupancy_grid_steps", [-1]
        )
        self.reinit_shape_every_n_steps = self.config.system.get(
            "reinit_shape_every_n_steps", -1
        )

    def forward(self, batch):
        return self.model(batch['rays'])

    def reinit_occupancy_grid(self):
        # Iteratie through dataset and reinitialize occupancy grid for each frame
        all_occs = []
        all_binaries = []
        all_aabbs = []
        dataloader = DataLoader(
            self.trainer.datamodule.train_dataloader().dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.rank)
            self.preprocess_data(batch, 'train')
            occs, binaries, aabb = self.model._compute_occupancy_grid(
                self.model.occupancy_grid.grid_coords, resolution=64
            )
            all_occs.append(occs)
            all_binaries.append(binaries)
            all_aabbs.append(aabb)

        occupancy_grid_old = self.model.occupancy_grid
        occupancy_grid_new = TemporalOccGridEstimator(
            roi_aabb=torch.stack(all_aabbs, dim=0),
            resolution=64,
            levels=len(all_aabbs),
        ).to(occupancy_grid_old.binaries.device)
        assert occupancy_grid_old.levels == 1
        assert occupancy_grid_old.cells_per_lvl == occupancy_grid_new.cells_per_lvl
        occupancy_grid_new.occs = torch.cat(all_occs, dim=0)
        occupancy_grid_new.binaries = torch.cat(all_binaries, dim=0)
        self.model.occupancy_grid = occupancy_grid_new

    def reinit_shape(self):
        self.model.deformer.set_iniialized(False)

    def preprocess_data(self, batch, stage):
        if "rgb" in batch:
            # It is possible that the dataset contains only poses and no images
            # (e.g. for animation datasets)
            rgb = batch["rgb"].reshape(-1, 3)
        if "valid_mask" in batch:
            valid_mask = batch["valid_mask"].reshape(-1)
        if "albedo" in batch:
            albedo = batch["albedo"].reshape(-1, 3)
        if "normal" in batch:
            normal = batch["normal"].reshape(-1, 3)
        if "hdri" in batch:
            assert stage in ["test"]
            assert batch["hdri"].shape[0] == 1
            batch["hdri"] = batch["hdri"].squeeze(0)

        rays = torch.cat(
            [
                batch["rays_o"],
                batch["rays_d"],
                batch["near"][..., None],
                batch["far"][..., None],
            ],
            dim=-1,
        ).reshape(-1, 8)
        batch.pop("rays_o")
        batch.pop("rays_d")
        batch.pop("near")
        batch.pop("far")

        self.train_num_rays = rays.shape[0]

        batch.update({"rays": rays})

        if stage in ["train"]:
            if self.config.model.background_color == "white":
                self.model.background_color = torch.ones(
                    (3,), dtype=torch.float32, device=self.rank
                )
            elif self.config.model.background_color == "black":
                self.model.background_color = torch.zeros(
                    (3,), dtype=torch.float32, device=self.rank
                )
            elif self.config.model.background_color == "random":
                self.model.background_color = torch.rand(
                    (3,), dtype=torch.float32, device=self.rank
                )
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones(
                (3,), dtype=torch.float32, device=self.rank
            )

        if "rgb" in batch:
            fg_mask = batch["alpha"].reshape(-1)
            rgb_wo_mask = rgb
            rgb = rgb * fg_mask[..., None] + rgb_to_srgb(
                self.model.background_color * (1 - fg_mask[..., None])
            )
            batch.update({"rgb": rgb, "rgb_wo_mask": rgb_wo_mask, "alpha": fg_mask})

        if "valid_mask" in batch:
            batch.update({"valid_mask": valid_mask})
        if "albedo" in batch:
            batch.update({"albedo": albedo})
        if "normal" in batch:
            batch.update({"normal": normal})

        if stage in ["train"]:
            self.model.t_idx = batch["t_idx"]
        else:
            self.model.t_idx = 0.0  # t_idx is not used in val/test/predict

        self.model.prepare(batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0

        # Radiance field losses
        if (not self.config.system.pbr_loss_only) or (not self.model.enable_phys):
            loss_rgb_mse = F.mse_loss(
                out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
                batch["rgb"][out["rays_valid_full"][..., 0]],
            )
            self.log("train/loss_rgb_mse", loss_rgb_mse)
            loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

            loss_rgb_l1 = F.l1_loss(
                out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
                batch["rgb"][out["rays_valid_full"][..., 0]],
            )
            # loss_rgb_l1 = F.huber_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]], delta=0.1)
            self.log("train/loss_rgb", loss_rgb_l1)
            loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        # PBR losses
        if self.model.enable_phys and self.config.model.learn_material:
            if self.model.add_emitter:
                loss_rgb_phys_mse = F.mse_loss(
                    out["comp_rgb_phys_full"],
                    batch["rgb_wo_mask"],
                )
            else:
                loss_rgb_phys_mse = F.mse_loss(
                    out["comp_rgb_phys_full"][out["rays_valid_phys_full"][..., 0]],
                    batch["rgb"][out["rays_valid_phys_full"][..., 0]],
                )
            self.log("train/loss_rgb_phys_mse", loss_rgb_phys_mse)
            loss += loss_rgb_phys_mse * self.C(
                self.config.system.loss.lambda_rgb_phys_mse
            )

            if self.model.add_emitter:
                loss_rgb_phys_l1 = F.l1_loss(
                    out["comp_rgb_phys_full"],
                    batch["rgb_wo_mask"],
                )
            else:
                loss_rgb_phys_l1 = F.l1_loss(
                    out["comp_rgb_phys_full"][out["rays_valid_phys_full"][..., 0]],
                    batch["rgb"][out["rays_valid_phys_full"][..., 0]],
                )
            self.log("train/loss_rgb_phys", loss_rgb_phys_l1)
            loss += loss_rgb_phys_l1 * self.C(
                self.config.system.loss.lambda_rgb_phys_l1
            )

            # Demodulated losses from nvdiffrecmc (https://github.com/NVlabs/nvdiffrecmc)
            # this loss is set to 0 in the default model, we only use the corresponding output
            # as "shading" for visualization
            loss_rgb_demodulated = F.l1_loss(
                luma(out["comp_demod_phys_full"][out["rays_valid_phys_full"][..., 0]]),
                max_value(batch["rgb"][out["rays_valid_phys_full"][..., 0]]),
            )
            self.log("train/loss_rgb_demodulated", loss_rgb_demodulated)
            loss += loss_rgb_demodulated * self.C(
                self.config.system.loss.lambda_rgb_demodulated
            )

            if self.C(self.config.system.loss.lambda_albedo) > 0:
                loss_albedo = F.l1_loss(
                    out["comp_albedo_full"][out["rays_valid_phys_full"][..., 0]],
                    batch["albedo"][out["rays_valid_phys_full"][..., 0]],
                )
                self.log("train/loss_albedo", loss_albedo)
                loss += loss_albedo * self.C(self.config.system.loss.lambda_albedo)

        # Eikonal loss
        loss_eikonal = (
            (torch.linalg.norm(out["sdf_grad_samples"], ord=2, dim=-1) - 1.0) ** 2
        ).mean()
        self.log("train/loss_eikonal", loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        # Mask losses
        opacity = torch.clamp(out["opacity"].squeeze(-1), 1.0e-3, 1.0 - 1.0e-3)
        if self.dataset.has_mask:
            if self.C(self.config.system.loss.lambda_mask_mse) > 0:
                loss_mask_mse = F.mse_loss(opacity, batch["alpha"].float())
                self.log("train/loss_mask_mse", loss_mask_mse)
                loss += loss_mask_mse * self.C(self.config.system.loss.lambda_mask_mse)
            if self.C(self.config.system.loss.lambda_mask_bce) > 0:
                loss_mask_bce = binary_cross_entropy(opacity, batch["alpha"].float())
                self.log("train/loss_mask_bce", loss_mask_bce)
                loss += loss_mask_bce * self.C(self.config.system.loss.lambda_mask_bce)

        # Opaque loss
        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        # Sparsity loss
        loss_sparsity = torch.exp(
            -self.config.system.loss.sparsity_scale * out["sdf_samples"].abs()
        ).mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            loss_curvature = out["sdf_laplace_samples"].abs().mean()
            self.log("train/loss_curvature", loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(
                out["weights"], out["points"], out["intervals"], out["ray_indices"]
            )
            self.log("train/loss_distortion", loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        if (
            self.config.model.learned_background
            and self.C(self.config.system.loss.lambda_distortion_bg) > 0
        ):
            raise NotImplementedError("Learned background not implemented.")

        # Additional regularization losses
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f"train/loss_{name}", value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        # Also monitor beta values
        self.log("train/beta", out["beta"], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith("lambda"):
                self.log(f"train_params/{name}", self.C(value))

        self.log("train/num_rays", float(self.train_num_rays), prog_bar=True)

        return {"loss": loss}

    def transform_normals(self, batch, normals):
        """ Convert world-space normal map into OpenGL camera space
        """
        # Convert to camera space, if necessary
        if "w2c" in batch:
            w2c = batch["w2c"]
            assert (w2c.shape[0] == 1) and (w2c.shape[1] == 4) and (w2c.shape[2] == 4)
            normals = torch.matmul(normals, w2c[0, :3, :3].transpose(0, 1))

        # Convert OpenCV to OpenGL convention
        normals = normals * torch.tensor([1.0, -1.0, -1.0], device=self.rank)

        return normals

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        ret = {}
        rf_psnr = self.criterions["psnr"](
            out["comp_rgb_full"].to(batch["rgb"]),
            batch["rgb"],
            valid_mask=batch["valid_mask"] if "valid_mask" in batch else None,
        )
        W, H = self.dataset.img_wh
        rf_ssim = self.criterions["ssim"](
            out["comp_rgb_full"].to(batch["rgb"]).reshape(H, W, 3),
            batch["rgb"].reshape(H, W, 3),
            valid_mask=batch["valid_mask"].reshape(H, W)
            if "valid_mask" in batch
            else None,
        )
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(batch["rgb"])
        rf_lpips = self.criterions["lpips"](
            out["comp_rgb_full"].to(batch["rgb"]).reshape(H, W, 3),
            batch["rgb"].reshape(H, W, 3),
            loss_fn_vgg,
            valid_mask=batch["valid_mask"].reshape(H, W)
            if "valid_mask" in batch
            else None,
        )
        ret.update({"rf_psnr": rf_psnr, "rf_ssim": rf_ssim, "rf_lpips": rf_lpips})
        # Convert normal map into OpenGL camera space
        out.update(
            {
                "comp_normal": self.transform_normals(
                    batch, out["comp_normal"].to(batch["rgb"])
                )
            }
        )
        if "normal" in batch:
            normal_error = self.criterions["normal_error"](
                F.normalize(out["comp_normal"].to(batch["normal"]), dim=-1),
                F.normalize(batch["normal"], dim=-1),
                valid_mask=(batch["alpha"] > 0.5),
            )
            assert torch.isfinite(normal_error)
            ret.update({"normal_error": normal_error})

        if self.model.enable_phys:
            pbr_psnr = self.criterions["psnr"](
                out["comp_rgb_phys_full"].to(batch["rgb"]),
                batch["rgb"],
                valid_mask=batch["valid_mask"] if "valid_mask" in batch else None,
            )
            pbr_ssim = self.criterions["ssim"](
                out["comp_rgb_phys_full"].to(batch["rgb"]).reshape(H, W, 3),
                batch["rgb"].reshape(H, W, 3),
                valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
            )
            pbr_lpips = self.criterions["lpips"](
                out["comp_rgb_phys_full"].to(batch["rgb"]).reshape(H, W, 3),
                batch["rgb"].reshape(H, W, 3),
                loss_fn_vgg,
                valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
            )
            ret.update(
                {"pbr_psnr": pbr_psnr, "pbr_ssim": pbr_ssim, "pbr_lpips": pbr_lpips}
            )
            if "albedo" in batch:
                # Both GT and predicted albedo are in linear RGB space
                gt_albedo = batch["albedo"]
                pred_albedo = out["comp_albedo_full"].to(gt_albedo)
                gt_mask = batch["alpha"] > 0.5

                # Align predicted albedo with linear RGB GT
                three_channel_ratio = []
                for i in range(gt_albedo.shape[-1]):
                    x = gt_albedo[gt_mask][:, i]
                    x_hat = pred_albedo[gt_mask][:, i]
                    scale = torch.sum(x * x_hat) / torch.sum(x_hat * x_hat)
                    three_channel_ratio.append(scale)

                three_channel_ratio = torch.stack(three_channel_ratio, dim=0)

                three_aligned_albedo = torch.zeros_like(gt_albedo)
                three_aligned_albedo[gt_mask] = (
                    three_channel_ratio * pred_albedo[gt_mask]
                ).clamp(min=0.0, max=1.0)

                three_aligned_psnr = self.criterions["psnr"](
                    three_aligned_albedo, gt_albedo, valid_mask=gt_mask
                )
                three_aligned_ssim = self.criterions["ssim"](
                    three_aligned_albedo.reshape(H, W, 3),
                    gt_albedo.reshape(H, W, 3),
                    valid_mask=gt_mask.reshape(H, W),
                )
                three_aligned_lpips = self.criterions["lpips"](
                    three_aligned_albedo.reshape(H, W, 3),
                    gt_albedo.reshape(H, W, 3),
                    loss_fn_vgg,
                    valid_mask=gt_mask.reshape(H, W),
                )
                ret.update(
                    {
                        "albedo_psnr": three_aligned_psnr,
                        "albedo_ssim": three_aligned_ssim,
                        "albedo_lpips": three_aligned_lpips,
                    }
                )

        imgs = self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0].item()}.png",
            # GT and radiance field rendering
            [
                {
                    "type": "rgb",
                    "img": batch["rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_rgb_full"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            # PBR rendering
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_phys_full"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_demod_phys_full"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": three_aligned_albedo.view(H, W, 3)
                        if "albedo" in batch
                        else out["comp_albedo_full"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_metallic_full"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_roughness_full"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                ]
                if self.model.enable_phys
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["visibility"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                ]
                if self.config.model.render_mode == "uniform_light"
                and self.model.enable_phys
                else []
            )
            # Background rendering
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_bg"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if self.config.model.learned_background
                else []
            )
            # Depth and normal
            + [
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "rgb",
                    "img": out["comp_normal"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                },
            ],
        )
        # Also log images to wandb/tensorboard
        captions = (
            ["gt_rgb", "rf_rgb"]
            + (
                [
                    "pbr_rgb",
                    "pbr_demod",
                    "pbr_albedo",
                    "pbr_metallic",
                    "pbr_roughness",
                ]
                if self.model.enable_phys
                else []
            )
            + (
                ["visibility"]
                if self.config.model.render_mode == "uniform_light"
                and self.model.enable_phys
                else []
            )
            + (
                ["bg_rgb", "bg_rgb_wo_mask"]
                if self.config.model.learned_background
                else []
            )
            + ["depth", "normal"]
        )
        # Convert bgr2rgb
        imgs = [img[:, :, [2, 1, 0]] for img in imgs]
        self.logger.log_image(
            key="validation_samples", images=imgs, caption=captions
        )

        ret.update({'index': batch['index']})
        return ret

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        # Save environment map
        if hasattr(self.model, 'emitter'):
            env_map = self.model.emitter.generate_image()
            self.save_image_grid(f"it{self.global_step}-envmap.exr", [
                {'type': 'hdr', 'img': env_map, 'kwargs': {'data_format': 'HWC'}},
            ])

        # Save occupancy grid
        if hasattr(self.model, 'occupancy_grid'):
            occ_grid = self.model.occupancy_grid.binaries
            self.save_data(f"it{self.global_step}-occgrid.npy", occ_grid)

        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    # out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    # Metrics
                    out_set[step_out["index"].item()] = {
                        k: v
                        for k, v in step_out.items()
                        if k not in ["index"]
                    }
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        # out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        out_set[index[0].item()] = {
                            k: v[oi]
                            for k, v in step_out.items()
                            if k not in ["index"]
                        }

            for metric_name in list(out_set.values())[0].keys():
                metric = torch.mean(
                    torch.stack([o[metric_name] for o in out_set.values()])
                )
                self.log(
                    f"val/{metric_name}", metric, prog_bar=True, rank_zero_only=True
                )

    def test_step(self, batch, batch_idx):
        if "hdri" in batch and "albedo" in batch:
            # If both HDRI and GT albedo are available, we can first align the predicted albedo
            # with the GT albedo via a partial forward pass. Relighting will also use aligned albedo.
            self.model.albedo_only = True
            gt_albedo = batch["albedo"]
            gt_mask = batch["alpha"] > 0.5
            # Fast inference to get albedo
            pred_albedo = self(batch)["comp_albedo_full"].to(gt_albedo)
            three_channel_ratio = compute_albedo_rescale_factor(
                gt_albedo, pred_albedo, gt_mask
            )
            self.model.albedo_align_ratio = three_channel_ratio
            self.model.albedo_only = False
            albedo_aligned = True
        else:
            albedo_aligned = False

        out = self(batch)
        if hasattr(self.model, 'albedo_align_ratio'):
            del self.model.albedo_align_ratio

        W, H = self.dataset.img_wh
        ret = {}
        if 'rgb' in batch:
            loss_fn_vgg = lpips.LPIPS(net="vgg").to(batch["rgb"])
            rf_psnr = self.criterions["psnr"](
                out["comp_rgb_full"].to(batch["rgb"]),
                batch["rgb"],
                valid_mask=batch["valid_mask"] if "valid_mask" in batch else None,
            )
            rf_ssim = self.criterions["ssim"](
                out["comp_rgb_full"].to(batch["rgb"]).reshape(H, W, 3),
                batch["rgb"].reshape(H, W, 3),
                valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
            )
            rf_lpips = self.criterions["lpips"](
                out["comp_rgb_full"].to(batch["rgb"]).reshape(H, W, 3),
                batch["rgb"].reshape(H, W, 3),
                loss_fn_vgg,
                valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
            )
            ret.update({"rf_psnr": rf_psnr, "rf_ssim": rf_ssim, "rf_lpips": rf_lpips})
        # Convert normal map into OpenGL camera space
        out.update(
            {
                "comp_normal": self.transform_normals(
                    batch,
                    out["comp_normal"].to(
                        batch["rgb"] if "rgb" in batch else self.rank
                    ),
                )
            }
        )
        if "normal" in batch:
            normal_error = self.criterions["normal_error"](
                F.normalize(out["comp_normal"].to(batch["normal"]), dim=-1),
                F.normalize(batch["normal"], dim=-1),
                valid_mask=(batch["alpha"] > 0.5),
            )
            assert torch.isfinite(normal_error)
            ret.update({"normal_error": normal_error})

        if self.model.enable_phys:
            if 'rgb' in batch:
                pbr_psnr = self.criterions["psnr"](
                    out["comp_rgb_phys_full"].to(batch["rgb"]),
                    batch["rgb"],
                    valid_mask=batch["valid_mask"] if "valid_mask" in batch else None,
                )
                pbr_ssim = self.criterions["ssim"](
                    out["comp_rgb_phys_full"].to(batch["rgb"]).reshape(H, W, 3),
                    batch["rgb"].reshape(H, W, 3),
                    valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
                )
                pbr_lpips = self.criterions["lpips"](
                    out["comp_rgb_phys_full"].to(batch["rgb"]).reshape(H, W, 3),
                    batch["rgb"].reshape(H, W, 3),
                    loss_fn_vgg,
                    valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
                )
                ret.update(
                    {"pbr_psnr": pbr_psnr, "pbr_ssim": pbr_ssim, "pbr_lpips": pbr_lpips}
                )
            if "albedo" in batch:
                if not albedo_aligned:
                    # If albedo prediction is not already aligned with GT, we need to align it here
                    # Both GT and predicted albedo are in linear RGB space
                    gt_albedo = batch["albedo"]
                    pred_albedo = out["comp_albedo_full"].to(gt_albedo)
                    gt_mask = batch["alpha"] > 0.5

                    three_channel_ratio = compute_albedo_rescale_factor(
                        gt_albedo, pred_albedo, gt_mask
                    )

                    three_aligned_albedo = torch.zeros_like(gt_albedo)
                    three_aligned_albedo[gt_mask] = (
                        three_channel_ratio * pred_albedo[gt_mask]
                    ).clamp(min=0.0, max=1.0)
                else:
                    three_aligned_albedo = out["comp_albedo_full"].to(batch["albedo"])

                three_aligned_psnr = self.criterions["psnr"](
                    three_aligned_albedo, gt_albedo, valid_mask=gt_mask
                )
                three_aligned_ssim = self.criterions["ssim"](
                    three_aligned_albedo.reshape(H, W, 3),
                    gt_albedo.reshape(H, W, 3),
                    valid_mask=gt_mask.reshape(H, W),
                )
                three_aligned_lpips = self.criterions["lpips"](
                    three_aligned_albedo.reshape(H, W, 3),
                    gt_albedo.reshape(H, W, 3),
                    loss_fn_vgg,
                    valid_mask=gt_mask.reshape(H, W),
                )
                ret.update(
                    {
                        "albedo_psnr": three_aligned_psnr,
                        "albedo_ssim": three_aligned_ssim,
                        "albedo_lpips": three_aligned_lpips,
                    }
                )

        # Save image grid
        imgs = self.save_image_grid(
            f"it{self.global_step}-test-all/{batch['index'][0].item()}.png",
            # GT image
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "rgb" in batch
                else []
            )
            # Radiance field rendering
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb_full"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            # PBR rendering
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_phys_full"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_demod_phys_full"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": three_aligned_albedo.view(H, W, 3)
                        if "albedo" in batch
                        else out["comp_albedo_full"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_metallic_full"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_roughness_full"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                ]
                if self.model.enable_phys
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["visibility"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                ]
                if self.config.model.render_mode == "uniform_light"
                and self.model.enable_phys
                else []
            )
            # Background rendering
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_bg"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if self.config.model.learned_background
                else []
            )
            # Depth and normal
            + [
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "rgb",
                    "img": out["comp_normal"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                },
            ],
        )
        # Also save images individually
        captions = (
            (["gt_rgb"] if "rgb" in batch else [])
            + ["rf_rgb"]
            + (
                [
                    "pbr_rgb",
                    "pbr_demod",
                    "pbr_albedo",
                    "pbr_metallic",
                    "pbr_roughness",
                ]
                if self.model.enable_phys
                else []
            )
            + (
                ["visibility"]
                if self.config.model.render_mode == "uniform_light"
                and self.model.enable_phys
                else []
            )
            + (
                ["bg_rgb", "bg_rgb_wo_mask"]
                if self.config.model.learned_background
                else []
            )
            + ["depth", "normal"]
        )
        for img, caption in zip(imgs, captions):
            self.save_image(
                    f"it{self.global_step}-test/{batch['index'][0].item():04}-{caption}.png",
                img[:, :, [2, 1, 0]],
            )

        # Also save images individually, with either predicted or ground-truth alpha mask
        for img, caption in zip(imgs, captions):
            if "gt" in caption:
                alpha = batch["alpha"].view(H, W).clamp(0, 1).cpu().numpy()
            else:
                alpha = out["opacity"].view(H, W).clamp(0, 1).cpu().numpy()

            alpha = (alpha * 255).astype(np.uint8)
            img = np.concatenate([img[:, :, [2, 1, 0]], alpha[:, :, None]], axis=-1)
            self.save_image(
                f"it{self.global_step}-test-with-alpha/{batch['index'][0].item():04}-{caption}.png",
                img,
            )

        ret.update({'index': batch['index']})
        return ret

    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        # Save environment map
        if hasattr(self.model, 'emitter'):
            env_map = self.model.emitter.generate_image()
            self.save_image_grid(f"it{self.global_step}-envmap.exr", [
                {'type': 'hdr', 'img': env_map, 'kwargs': {'data_format': 'HWC'}},
            ])

        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    # out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    out_set[step_out["index"].item()] = {
                        k: v
                        for k, v in step_out.items()
                        if k not in ["index"]
                    }
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        # out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        out_set[index[0].item()] = {
                            k: v[oi]
                            for k, v in step_out.items()
                            if k not in ["index"]
                        }

            # psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            # self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)
            for metric_name in list(out_set.values())[0].keys():
                metric = torch.mean(
                    torch.stack([o[metric_name] for o in out_set.values()])
                )
                self.log(
                    f"test/{metric_name}", metric, prog_bar=True, rank_zero_only=True
                )
                if metric_name == "albedo_psnr" and wandb.run is not None:
                    wandb.run.notes = f"albedo PSNR: {metric.item():.2f}"

            # self.save_img_sequence(
            #     f"it{self.global_step}-test",
            #     f"it{self.global_step}-test",
            #     '(\d+)\.png',
            #     save_format='mp4',
            #     fps=30
            # )

            self.export()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh,
        )
