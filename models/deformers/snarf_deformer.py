from .fast_snarf.deformer_torch import ForwardDeformer
from .smplx import SMPL
import torch
import torch.nn.functional as F

from torchgeometry.core import conversions


def get_predefined_rest_pose(cano_pose, device="cuda"):
    body_pose_t = torch.zeros((1, 69), device=device)
    if cano_pose.lower() == "da_pose":
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
    elif cano_pose.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    else:
        raise ValueError("Unknown cano_pose: {}".format(cano_pose))
    return body_pose_t


def get_bbox_from_smpl(vs, factor=1.2):
    assert vs.shape[0] == 1
    min_vert = vs.min(dim=1).values
    max_vert = vs.max(dim=1).values

    c = (max_vert + min_vert) / 2
    s = (max_vert - min_vert) / 2
    s = s.max(dim=-1).values * factor

    min_vert = c - s[:, None]
    max_vert = c + s[:, None]
    return torch.cat([min_vert, max_vert], dim=0)


class SNARFDeformer():
    # def __init__(self, model_path, gender, opt) -> None:
    def __init__(self, config) -> None:
        self.body_model = SMPL(config.model_path, gender=config.gender)
        self.deformer = ForwardDeformer(config.deformer_config)
        self.initialized = False
        self.opt = config.deformer_config

    def initialize(self, betas, device):
        if isinstance(self.opt.cano_pose, str):
            body_pose_t = get_predefined_rest_pose(self.opt.cano_pose, device=device)
        else:
            body_pose_t = torch.zeros((1, 69), device=device)
            body_pose_t[:, 2] = self.opt.cano_pose[0]
            body_pose_t[:, 5] = self.opt.cano_pose[1]
            body_pose_t[:, 47] = self.opt.cano_pose[2]
            body_pose_t[:, 50] = self.opt.cano_pose[3]

        smpl_outputs = self.body_model(betas=betas[:1], body_pose=body_pose_t)
        self.tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        self.vs_template = smpl_outputs.vertices
        part_ids_verts = self.body_model.lbs_weights.argmax(dim=-1)
        part_ids_faces = part_ids_verts[self.body_model.faces_tensor]
        self.faces_to_include = (part_ids_faces < 20).any(
            -1
        )  # exclude hand faces when computing inside losses

        # initialize SNARF
        self.deformer.device = device
        self.deformer.switch_to_explicit(resolution=self.opt.resolution,
                                         smpl_verts=smpl_outputs.vertices.float().detach(),
                                         smpl_weights=self.body_model.lbs_weights.clone()[None].detach(),
                                         use_smpl=True)
        self.bbox = get_bbox_from_smpl(smpl_outputs.vertices.detach())

        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)

    def prepare_deformer(self, smpl_params):
        device = smpl_params["betas"].device
        if self.opt.optimize_betas:
            betas = smpl_params["betas"] + smpl_params["betas_correction"]
        else:
            betas = smpl_params["betas"]
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
        if not self.initialized:
            self.initialize(betas.detach(), device)
            self.initialized = True

        body_pose = smpl_params["body_pose"] + smpl_params["pose_correction"]
        global_orient = (
            smpl_params["global_orient"] + smpl_params["global_orient_correction"]
        )
        transl = smpl_params["transl"] + smpl_params["transl_correction"]
        smpl_outputs = self.body_model(
            betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl
        )
        s2w = smpl_outputs.A[:, 0].float()
        w2s = torch.inverse(s2w)

        tfs = (w2s[:, None] @ smpl_outputs.A.float() @ self.tfs_inv_t).type(self.dtype)
        self.deformer.precompute(tfs)

        self.w2s = w2s
        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[
            :, None, :3, 3
        ]
        self.basic_joints = (
            smpl_outputs.joints[:, :24] @ w2s[:, :3, :3].permute(0, 2, 1)
        ) + w2s[:, None, :3, 3]
        rot_mats = conversions.angle_axis_to_rotation_matrix(
            smpl_params["body_pose"].reshape(-1, 3)
        )[:, :3, :3].reshape(-1, 23, 3, 3)
        self.rot_mats = torch.cat(
            [
                torch.eye(3, device=device)[None].repeat(rot_mats.shape[0], 1, 1, 1),
                rot_mats,
            ],
            dim=1,
        ).reshape(-1, 24, 9)
        self.tfs = tfs
        self.smpl_outputs = smpl_outputs
        self.smpl_params = smpl_params

    def transform_rays_w2s(self, rays):
        """transform rays from world to smpl coordinate system"""
        w2s = self.w2s.detach()
        assert (w2s.shape[0] == 1)

        # rays[:, :3] = (rays[:, :3] @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        # rays[:, 3:6] = (rays[:, 3:6] @ w2s[:, :3, :3].permute(0, 2, 1)).to(rays)
        # d = torch.norm(rays[:, :3], dim=-1)
        # rays[:, 6] = d - 1
        # rays[:, 7] = d + 1
        rays_o = (rays[:, :3] @ w2s[0, :3, :3].permute(1, 0)) + w2s[0, None, :3, 3]
        rays_d = rays[:, 3:6] @ w2s[0, :3, :3].permute(1, 0)
        d = torch.linalg.norm(rays_o, dim=-1, keepdim=True)
        near = d - 1
        far = d + 1

        return torch.cat([rays_o, rays_d, near, far], dim=-1)

    def transform_dirs_w2s(self, d):
        """transform directions from world to smpl coordinate system"""
        w2s = self.w2s.detach()
        assert (w2s.shape[0] == 1)
        return F.normalize(
            (d @ w2s[0, :3, :3].permute(1, 0)).to(d), dim=-1, eps=1e-6
        )

    def transform_dirs_s2w(self, d):
        """transform directions from smpl to world coordinate system"""
        w2s = self.w2s.detach()
        assert (w2s.shape[0] == 1)
        return F.normalize((d @ w2s[0, :3, :3]).to(d), dim=-1, eps=1e-6)

    def transform_rots_s2w(self, J_inv):
        """apply smpl->world rotation to J_inv"""
        w2s = self.w2s.detach()
        return (w2s[:, :3, :3].permute(0, 2, 1) @ J_inv).to(J_inv)

    def get_bbox_deformed(self):
        voxel = self.deformer.voxel_d[0].reshape(3, -1)
        return [voxel.min(dim=1).values, voxel.max(dim=1).values]

    def deform_(self, pts, eval_mode):
        """transform pts to canonical space"""
        point_size = pts.shape[0]
        betas = self.smpl_outputs.betas
        batch_size = betas.shape[0]
        pts = pts.reshape(batch_size, -1, 3)

        pts_cano, others = self.deformer.forward(pts, cond=None, tfs=self.tfs, eval_mode=eval_mode)
        valid = others["valid_ids"].reshape(point_size, -1)
        if self.opt.use_j_inv:
            # Use the (approximated) inverse Jacobian to transform normals
            J_inv = others["J_inv"].reshape(point_size, -1, 3, 3)
        else:
            # Use the linearly blended bone transforms to transform normals
            J_inv = others["fwd_tfs"].reshape(point_size, -1, 3, 3)

        return pts_cano.reshape(point_size, -1, 3), J_inv, valid

    def deform(self, pts, model, eval_mode):
        pts_cano_all, J_inv, valid = self.deform_(pts.type(self.dtype), eval_mode=eval_mode)

        c2w = J_inv[valid]

        sdf_cano = torch.ones_like(pts_cano_all[..., 0]).float() * 1e5  # canonical SDF
        sdf_grad = None # SDF gradient in observation space
        sdf_grad_cano = None    # SDF gradient in canonical space
        features = None # features from the canonical SDF field
        laplace = None  # Laplacian of the canonical SDF field

        if not torch.isfinite(pts_cano_all).all():
            print("WARNING: NaN found in pts_cano_all")
        # Note that model should also take care of the case where input Tensor is empty
        # (i.e. all points are invalid), and return empty tensors
        model_ret, J_inv_nr = model(pts_cano_all[valid])
        # if J_inv_nr is not None:
        c2w = c2w @ J_inv_nr
        any_valid = valid.any()
        if isinstance(model_ret, tuple) or isinstance(model_ret, list):
            if any_valid:
                sdf_cano[valid] = model_ret[0]

            if len(model_ret) > 1:
                sdf_grad_cano = torch.tensor(
                    [0, 0, 1], dtype=torch.float32, device=pts_cano_all.device
                ).repeat((*pts_cano_all.shape[:2], 1))
                if any_valid:
                    sdf_grad_cano[valid] = model_ret[1]
                assert c2w is not None
                sdf_grad = torch.tensor(
                    [0, 0, 1], dtype=torch.float32, device=pts_cano_all.device
                ).repeat((*pts_cano_all.shape[:2], 1))
                if any_valid:
                    sdf_grad[valid] = torch.einsum(
                        "bij,bj->bi", c2w, sdf_grad_cano[valid]
                    )
            if len(model_ret) > 2:
                features = torch.zeros(
                    (*pts_cano_all.shape[:2], model_ret[2].size(-1)),
                    dtype=torch.float32,
                    device=pts_cano_all.device,
                )
                if any_valid:
                    features[valid] = model_ret[2]
            if len(model_ret) > 3:
                laplace = torch.zeros_like(pts_cano_all[..., 0]).float()
                if any_valid:
                    laplace[valid] = model_ret[3]
        elif isinstance(model_ret, torch.Tensor):
            if any_valid:
                sdf_cano[valid] = model_ret
        else:
            raise ValueError("Invalid return type from model")

        sdf_cano, idx = torch.min(sdf_cano, dim=-1)
        pts_cano = torch.gather(pts_cano_all, 1, idx[:, None, None].repeat(1, 1, 3)).reshape(-1, 3)
        valid_cano = valid.any(dim=-1)
        out = [pts_cano, sdf_cano, valid_cano]
        if sdf_grad is not None:
            sdf_grad = torch.gather(sdf_grad, 1, idx[:, None, None].repeat(1, 1, 3)).reshape(-1, 3)
            out.append(sdf_grad)
        if sdf_grad_cano is not None:
            sdf_grad_cano = torch.gather(sdf_grad_cano, 1, idx[:, None, None].repeat(1, 1, 3)).reshape(-1, 3)
            out.append(sdf_grad_cano)
        if features is not None:
            features = torch.gather(
                features, 1, idx[:, None, None].repeat(1, 1, features.size(-1))
            ).reshape(-1, features.size(-1))
            out.append(features)
        if laplace is not None:
            laplace = torch.gather(laplace, 1, idx[:, None]).reshape(-1)
            out.append(laplace)

        return out

    def __call__(self, pts, model, eval_mode=True):
        return self.deform(pts, model, eval_mode)
