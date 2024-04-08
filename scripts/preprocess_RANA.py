import os
import torch

import trimesh
import cv2
import glob
import re
import json
import shutil
import argparse

import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from scipy.spatial.transform import Rotation as Rotation

# body models
from models.deformers.smplx import SMPL
from models.deformers.snarf_deformer import get_predefined_rest_pose
from models.utils import get_perspective

parser = argparse.ArgumentParser(description="Preprocessing for RANA.")
parser.add_argument(
    "--data-dir", type=str, help="Directory that contains raw RANA data."
)
parser.add_argument(
    "--split",
    type=str,
    help="Split to process.",
    choices=["train_p1", "test"],
)
parser.add_argument(
    "--out-dir", type=str, help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--seqname", type=str, default="subject_01", help="Sequence to process."
)
# To use the `--visualize` option, intall the following packages:
# `sudo apt-get install libosmesa6-dev` # for Ubuntu only
# `pip install pyrender`
# `pip install pyopengl==3.1.4`
# For an OS other than Ubuntu you may need to build OSMesa from source.
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")

if __name__ == "__main__":
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, args.split, seq_name)
    out_dir = os.path.join(args.out_dir, args.split, seq_name)

    os.makedirs(out_dir, exist_ok=True)
    shape = []
    global_orient = []
    body_pose = []
    transl = []
    vertices_smpl_space = []

    if args.split == "test":
        # Make directory for HDR maps
        hdri_dir = os.path.join(args.out_dir, "hdri")
        os.makedirs(hdri_dir, exist_ok=True)
        hdri_files = []

    img_pattern = re.compile(r"frame_(\d{6})\.png$")
    img_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.png")))
    img_files = [f for f in img_files if img_pattern.match(os.path.basename(f))]

    # Load gender information
    base_name = os.path.basename(img_files[0]).split(".")[0]
    json_file = os.path.join(data_dir, base_name + ".json")
    with open(json_file, 'r') as f:
        annots = json.load(f)

    # Standard SMPL model
    gender = np.array(annots['skeleton_0']['smpl_data']['gender']).tolist()
    body_model_smpl = SMPL(model_path="./data/SMPLX/smpl", gender=gender).cuda()

    if args.visualize:
        mesh_dir = os.path.join(out_dir, "smpl_vis", "meshes")
        vis_dir = os.path.join(out_dir, "smpl_vis", "images")
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    img_out_dir = os.path.join(out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    albedo_out_dir = os.path.join(out_dir, "albedos")
    os.makedirs(albedo_out_dir, exist_ok=True)
    normal_out_dir = os.path.join(out_dir, "normals")
    os.makedirs(normal_out_dir, exist_ok=True)
    mask_out_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_out_dir, exist_ok=True)

    device = torch.device("cuda")

    for img_idx, img_file in enumerate(img_files):
        print("Processing: {}".format(img_file))

        base_name = os.path.basename(img_file).split(".")[0]
        json_file = os.path.join(data_dir, base_name + ".json")
        with open(json_file, "r") as f:
            annots = json.load(f)

        if args.split == "test":
            assert "bg_file" in annots.keys()
            assert "yaw" in annots['camera']
            assert "fov" in annots['camera']
            # Download and save HDR map
            url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/{}".format(
                annots["bg_file"]
            )
            hdri_file = os.path.join(hdri_dir, os.path.basename(url))
            if not os.path.exists(hdri_file):
                os.system("wget {} -P {}".format(url, hdri_dir))

            hdri_files.append(os.path.basename(hdri_file))

            # # Load HDR image and get height/width
            # hdr_img = cv2.imread(hdri_file, cv2.IMREAD_COLOR)
            # hdr_height, hdr_width = hdr_img.shape[:2]

            yaw = annots['camera']['yaw']
            assert yaw == 0
            theta = -270 - yaw
            intrinsic, R = get_perspective(
                fov=np.rad2deg(annots['camera']['fov']),
                theta=theta,
                phi=0,
                height=720,
                width=1280,
            )
        else:
            # Intrinsics
            intrinsic = np.array(annots['skeleton_0']['smpl_data']['K'])
            # Extrinsics
            R = np.eye(3, dtype=np.float32)

        T = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        extrinsic = np.block([[R, T], [0, 0, 0, 1]])

        smpl_data = annots['skeleton_0']['smpl_data']
        # verts_smpl = np.array(smpl_data['vertices']) / 1000.0
        smpl_pose = np.array(smpl_data["pose"], dtype=np.float32).reshape(1, -1)
        smpl_pose[:, 57:] = 0.0    # set hand pose to zero
        smpl_betas = np.array(smpl_data["betas"], dtype=np.float32).reshape(1, -1)
        smpl_global_orient = np.array(
            smpl_data["global_orient"], dtype=np.float32
        ).reshape(1, -1)
        global_trans = np.array(
            smpl_data["global_trans"], dtype=np.float32
        ).reshape(3, 1)
        global_scale = np.array(smpl_data["scale"], dtype=np.float32)

        poses_torch = torch.from_numpy(smpl_pose).cuda()
        betas_torch = torch.from_numpy(smpl_betas).cuda()
        global_orient_torch = torch.from_numpy(smpl_global_orient).cuda()

        out = body_model_smpl(
            betas=betas_torch, body_pose=poses_torch, global_orient=global_orient_torch
        )
        smpl_p3d = out.joints
        smpl_root = smpl_p3d[:, :1]
        # verts_rel = (out.vertices - smpl_root)[0].detach().cpu().numpy()
        # verts_smpl = (verts_rel * global_scale) + global_trans

        smpl_transl = (
            -smpl_root[0].detach().cpu().numpy()
            + global_trans.reshape(1, -1) / global_scale
        )
        transl_torch = torch.from_numpy(smpl_transl.reshape(1, -1)).cuda()

        # Record SMPL parameters
        if img_idx == 0:
            shape = smpl_betas.copy()
            body_pose_t = get_predefined_rest_pose("a_pose", device=device)
            smpl_outputs = body_model_smpl(betas=betas_torch, body_pose=body_pose_t)
            tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        else:
            assert (shape.flatten() == smpl_betas.flatten()).all()

        if args.split == "test":
            # Re-compute SMPL mesh with new translation
            smpl_outputs = body_model_smpl(
                betas=betas_torch,
                body_pose=poses_torch,
                global_orient=global_orient_torch,
                transl=transl_torch,
            )
            # Apply inverse R to smpl_outputs.vertices
            vertices_tgt = smpl_outputs.vertices[0].detach().cpu().numpy()
            vertices_tgt = vertices_tgt @ R
            # Compute new smpl_global_orient and global_orient_torch
            smpl_global_orient = (
                R.T @ Rotation.from_rotvec(smpl_global_orient).as_matrix()
            )
            smpl_global_orient = (
                Rotation.from_matrix(smpl_global_orient).as_rotvec().astype(np.float32)
            )
            global_orient_torch = torch.from_numpy(smpl_global_orient).cuda().float()
            # Re-compute SMPL mesh with new orientation
            smpl_outputs = body_model_smpl(
                betas=betas_torch,
                body_pose=poses_torch,
                global_orient=global_orient_torch,
                transl=transl_torch,
            )

            # Compute new smpl_transl and transl_torch
            verticies = smpl_outputs.vertices[0].detach().cpu().numpy()
            smpl_transl = smpl_transl + (vertices_tgt - verticies).mean(0, keepdims=True)
            transl_torch = torch.from_numpy(smpl_transl.reshape(1, -1)).cuda()

        global_orient.append(smpl_global_orient)
        body_pose.append(smpl_pose)
        transl.append(smpl_transl)

        # Visualize SMPL mesh
        if args.visualize:
            from utils.smpl_renderer import Renderer
            # Re-compute SMPL mesh with new translation
            smpl_outputs = body_model_smpl(
                betas=betas_torch,
                body_pose=poses_torch,
                global_orient=global_orient_torch,
                transl=transl_torch,
            )
            # Vertices in world coordinate
            verts_smpl = smpl_outputs["vertices"][0].detach().cpu().numpy()

            # Also record vertices in SMPL pelvis-aligned coordinate
            s2w = smpl_outputs.A[:, 0].float()
            w2s = torch.inverse(s2w)

            tfs = w2s[:, None] @ smpl_outputs.A.float() @ tfs_inv_t

            vertices_s = (
                smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)
            ) + w2s[:, None, :3, 3]
            vertices_s = vertices_s[0].detach().cpu().numpy()   # pelvis-aligned coordinate
            vertices_smpl_space.append(vertices_s)
            # Save mesh as .obj
            mesh = trimesh.Trimesh(
                vertices=vertices_s, faces=body_model_smpl.faces
            )
            mesh.export(os.path.join(mesh_dir, "{}.obj".format(base_name)))

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
                os.path.join(vis_dir, "{}.jpg".format(base_name)),
                cv2.cvtColor(smpl_image, cv2.COLOR_RGB2BGR),
            )

            del renderer

        shutil.copy(
            os.path.join(img_file),
            os.path.join(img_out_dir, "image_{:04d}.png".format(img_idx)),
        )
        albedo_file = os.path.join(data_dir, base_name + "_albedo.png")
        shutil.copy(
            os.path.join(albedo_file),
            os.path.join(albedo_out_dir, "albedo_{:04d}.png".format(img_idx)),
        )
        normal_file = os.path.join(data_dir, base_name + "_normals.png")
        shutil.copy(
            os.path.join(normal_file),
            os.path.join(normal_out_dir, "normal_{:04d}.png".format(img_idx)),
        )
        # load mask_file and save as .npy
        mask_file = os.path.join(data_dir, base_name + "_semantic.png")
        rgba = Image.open(mask_file)
        rgba = TF.to_tensor(rgba).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
        h, w = rgba.shape[:2]
        mask = (rgba[..., -1] > 0.5).byte().numpy()
        np.save(os.path.join(mask_out_dir, "mask_{:04d}.npy".format(img_idx)), mask)

        # Record camera parameters for this frame
        if img_idx == 0:
            cam_params = {
                "intrinsic": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
                "distortion": [0, 0, 0, 0],
                "height": h,
                "width": w,
            }
        else:
            assert cam_params["intrinsic"] == intrinsic.tolist()
            assert cam_params["extrinsic"] == extrinsic.tolist()
            assert cam_params["distortion"] == [0, 0, 0, 0]
            assert cam_params["height"] == h
            assert cam_params["width"] == w

    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cam_params, f)
    out_filename = os.path.join(out_dir, "poses.npz")
    np.savez(
        out_filename,
        betas=shape,
        global_orient=np.concatenate(global_orient, axis=0),
        body_pose=np.concatenate(body_pose, axis=0),
        transl=np.concatenate(transl, axis=0),
    )

    if args.split == "test":
        out_filename = os.path.join(out_dir, "hdri_files.json")
        with open(out_filename, "w") as f:
            json.dump(hdri_files, f)

    # # Iterate over all mesh vertices in vertices_smpl_space, compute aabb of all
    # # vertices
    # vertices_smpl_space = np.concatenate(vertices_smpl_space, axis=0)
    # aabb_min = vertices_smpl_space.min(0)
    # aabb_max = vertices_smpl_space.max(0)
    # aabb_center = (aabb_min + aabb_max) / 2.0
    # # aabb_size = aabb_max - aabb_min
    # # aabb_size = aabb_size.max() * 1.2
    # aabb_size = 2.5
    # aabb_min = aabb_center - aabb_size / 2.0
    # aabb_max = aabb_center + aabb_size / 2.0
    # aabb = np.concatenate([aabb_min, aabb_max], axis=0)
    # np.save(os.path.join(out_dir, "aabb.npy"), aabb)
