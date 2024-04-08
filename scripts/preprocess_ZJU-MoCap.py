import os
import torch

import cv2
import glob
import json
import shutil
import argparse

import numpy as np

# body models
from models.deformers.smplx import SMPL
from scripts.easymocap.smplmodel import SMPLlayer as SMPLlayerEM

from utils.smpl_renderer import Renderer

parser = argparse.ArgumentParser(description="Preprocessing for ZJU-MoCap.")
parser.add_argument(
    "--data-dir", type=str, help="Directory that contains raw ZJU-MoCap data."
)
parser.add_argument(
    "--out-dir", type=str, help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--seqname", type=str, default="CoreView_377", help="Sequence to process."
)
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


if __name__ == "__main__":
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)
    out_dir = os.path.join(args.out_dir, seq_name)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")

    if args.visualize:
        vis_dir = os.path.join(out_dir, "smpl_vis", "images")
        os.makedirs(vis_dir, exist_ok=True)

    vertices_out_dir = os.path.join(out_dir, "vertices")
    os.makedirs(vertices_out_dir, exist_ok=True)

    # Standard SMPL model
    # ZJU-MoCap uses gender-neutral SMPL model
    gender = "neutral"
    print(gender)

    body_model_smpl = SMPL(model_path="./data/SMPLX/smpl", gender=gender).cuda()

    # EasyMocap SMPL model
    body_model_em = SMPLlayerEM(
        "./data/SMPLX/smpl",
        model_type="smpl",
        gender=gender,
        device=device,
        regressor_path=os.path.join("data/smplh", "J_regressor_body25_smplh.txt"),
    )

    # Load annotations
    annots = np.load(os.path.join(data_dir, "annots.npy"), allow_pickle=True).item()
    motion_dir = os.path.join(data_dir, "new_params")
    cameras = annots["cams"]

    # Use only one camera for now
    cam_names = [1]

    if seq_name in ["CoreView_313", "CoreView_315"]:
        cam_names = ["Camera ({})".format(cam_name) for cam_name in cam_names]
    else:
        cam_names = ["Camera_B{}".format(cam_name) for cam_name in cam_names]

    all_cam_params = {"all_cam_names": cam_names}

    shape = []
    global_orient = []
    body_pose = []
    transl = []
    vertices_smpl_space = []

    for cam_idx, cam_name in enumerate(cam_names):
        intrinsic = np.array(cameras["K"][cam_idx])
        D = np.array(cameras["D"][cam_idx]).reshape([-1])
        R = np.array(cameras["R"][cam_idx])
        T = np.array(cameras["T"][cam_idx]) / 1000.0
        # R is 3x3, T is 3x3, construct 4x4 extrinsic matrix
        extrinsic = np.block([[R, T], [0, 0, 0, 1]])

        img_in_dir = os.path.join(data_dir, cam_name)
        mask_in_dir = os.path.join(data_dir, "mask_cihp/{}".format(cam_name))

        img_out_dir = os.path.join(out_dir, "images/{}".format(cam_name))
        os.makedirs(img_out_dir, exist_ok=True)
        mask_out_dir = os.path.join(out_dir, "masks/{}".format(cam_name))
        os.makedirs(mask_out_dir, exist_ok=True)
        bound_mask_out_dir = os.path.join(out_dir, "bound_masks/{}".format(cam_name))
        os.makedirs(bound_mask_out_dir, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(img_in_dir, "*.jpg")))

        cam_params = {
            "intrinsic": intrinsic.tolist(),
            "extrinsic": extrinsic.tolist(),
            "distortion": D.tolist(),
            "height": 1024,
            "width": 1024,
        }
        all_cam_params.update({cam_name: cam_params})

        if args.visualize:
            mesh_dir = os.path.join(out_dir, "smpl_vis", "meshes")
            vis_dir = os.path.join(out_dir, "smpl_vis", cam_name)
            os.makedirs(mesh_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)

        for img_idx, img_file in enumerate(img_files):
            print("Processing: {}".format(img_file))
            if seq_name in ["CoreView_313", "CoreView_315"]:
                idx = img_idx
            else:
                idx = int(os.path.basename(img_file)[:-4])

            mask_file = os.path.join(
                mask_in_dir, os.path.basename(img_file)[:-4] + ".png"
            )

            # We only process SMPL parameters in world coordinate
            if cam_idx == 0:
                param_idx = (idx + 1) if seq_name in ["CoreView_313", "CoreView_315"] else idx
                params = np.load(
                    os.path.join(motion_dir, "{}.npy".format(param_idx)), allow_pickle=True
                ).item()

                root_orient = np.array(params["Rh"], dtype=np.float32).reshape([1, 3])
                trans = np.array(params["Th"], dtype=np.float32).reshape([1, 3])

                betas = np.array(params["shapes"], dtype=np.float32)
                poses = np.array(params["poses"], dtype=np.float32)
                poses_smpl = poses[..., 3:].copy()

                betas_torch = torch.from_numpy(betas).cuda()
                poses_torch = torch.from_numpy(poses).cuda()
                poses_smpl_torch = torch.from_numpy(poses_smpl).cuda()

                root_orient_torch = torch.from_numpy(root_orient).cuda()
                trans_torch = torch.from_numpy(trans).cuda()

                device = torch.device("cuda")
                # Get mesh vertices from standard SMPL model
                smpl_outputs = body_model_smpl(
                    betas=betas_torch,
                    body_pose=poses_smpl_torch,
                    global_orient=root_orient_torch,
                    transl=trans_torch,
                )
                verts_smpl = smpl_outputs["vertices"][0].detach().cpu().numpy()

                # Get mesh vertices from EasyMocap SMPL model
                body_model_em = SMPLlayerEM(
                    "./data/SMPLX/smpl",
                    model_type="smpl",
                    gender=gender,
                    device=device,
                )
                verts = (
                    body_model_em(
                        poses=poses_torch,
                        shapes=betas_torch,
                        Rh=root_orient_torch,
                        Th=trans_torch,
                        return_verts=True,
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                new_trans = trans + (verts - verts_smpl).mean(0, keepdims=True)
                new_trans_torch = torch.from_numpy(new_trans).cuda()

                if img_idx == 0:
                    shape = betas.copy()
                    # body_pose_t = get_predefined_rest_pose("a_pose", device=device)
                    # smpl_outputs = body_model_smpl(betas=betas_torch, body_pose=body_pose_t)
                    # tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())

                global_orient.append(root_orient)
                body_pose.append(poses_smpl)
                transl.append(new_trans)

                # Visualize SMPL mesh
                if args.visualize:
                    # Re-compute SMPL mesh with new translation
                    smpl_outputs = body_model_smpl(
                        betas=betas_torch,
                        body_pose=poses_smpl_torch,
                        global_orient=root_orient_torch,
                        transl=new_trans_torch,
                    )
                    # Vertices for visualization
                    verts_smpl = smpl_outputs["vertices"][0].detach().cpu().numpy()

                    min_xyz = np.min(verts_smpl, axis=0)
                    max_xyz = np.max(verts_smpl, axis=0)
                    min_xyz -= 0.05
                    max_xyz += 0.05

                    bounds = np.stack([min_xyz, max_xyz], axis=0)
                    bound_mask = get_bound_2d_mask(
                        bounds, intrinsic, np.concatenate([R, T], axis=-1), 1024, 1024
                    )

                    cv2.imwrite(
                        os.path.join(
                            bound_mask_out_dir, "bound_mask_{:06d}.png".format(idx)
                        ),
                        bound_mask,
                    )

                    # # Also record vertices in SMPL canonical coordinate
                    # s2w = smpl_outputs.A[:, 0].float()
                    # w2s = torch.inverse(s2w)

                    # tfs = w2s[:, None] @ smpl_outputs.A.float() @ tfs_inv_t

                    # vertices_s = (
                    #     smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)
                    # ) + w2s[:, None, :3, 3]
                    # vertices_s = vertices_s[0].detach().cpu().numpy()
                    # vertices_smpl_space.append(vertices_s)
                    # # Save mesh as .obj
                    # mesh = trimesh.Trimesh(
                    #     vertices=vertices_s, faces=body_model_smpl.faces
                    # )
                    # mesh.export(os.path.join(mesh_dir, "{:06d}.obj".format(idx)))

                    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
                    renderer = Renderer(
                        height=img.shape[0],
                        width=img.shape[1],
                        faces=body_model_smpl.faces,
                    )
                    render_cameras = {
                        "K": [intrinsic],
                        "R": [np.array(R, dtype=np.float32)],
                        "T": [np.array(T, dtype=np.float32)],
                    }

                    render_data = {
                        0: {
                            "name": "SMPL",
                            "vertices": verts_smpl,
                            "faces": body_model_smpl.faces,
                            "vid": 2,
                        }
                    }

                    images = [img]
                    smpl_image = renderer.render(
                        render_data,
                        render_cameras,
                        images,
                        use_white=False,
                        add_back=True,
                    )[0]

                    cv2.imwrite(
                        os.path.join(vis_dir, "{:06d}.jpg".format(idx)),
                        cv2.cvtColor(smpl_image, cv2.COLOR_RGB2BGR),
                    )

                    del renderer

            shutil.copy(
                os.path.join(img_file),
                os.path.join(img_out_dir, "image_{:04d}.jpg".format(idx)),
            )
            # # load mask_file and save as .npy
            # mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # np.save(os.path.join(mask_out_dir, "mask_{:04d}.npy".format(idx)), mask)
            shutil.copy(
                os.path.join(mask_file),
                os.path.join(mask_out_dir, "mask_{:04d}.png".format(idx)),
            )

    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(all_cam_params, f)
    out_filename = os.path.join(out_dir, "poses.npz")
    np.savez(
        out_filename,
        betas=shape,
        global_orient=np.concatenate(global_orient, axis=0),
        body_pose=np.concatenate(body_pose, axis=0),
        transl=np.concatenate(transl, axis=0),
    )

    # # Iterate over all mesh vertices in vertices_smpl_space, compute aabb of all
    # # vertices, with padding of 10%, and save as .npy
    # vertices_smpl_space = np.concatenate(vertices_smpl_space, axis=0)
    # aabb_min = vertices_smpl_space.min(0)
    # aabb_max = vertices_smpl_space.max(0)
    # aabb_center = (aabb_min + aabb_max) / 2.0
    # aabb_size = aabb_max - aabb_min
    # aabb_size = aabb_size.max() * 1.2
    # aabb_min = aabb_center - aabb_size / 2.0
    # aabb_max = aabb_center + aabb_size / 2.0
    # aabb = np.concatenate([aabb_min, aabb_max], axis=0)
    # np.save(os.path.join(out_dir, "aabb.npy"), aabb)
