import os
import torch

# import trimesh
import cv2
import glob
import json
import pyexr
# import shutil
import argparse

import numpy as np

# body models
from models.deformers.smplx import SMPL
# from models.deformers.snarf_deformer import get_predefined_rest_pose
from scripts.easymocap.smplmodel import SMPLlayer as SMPLlayerEM

from utils.smpl_renderer import Renderer

parser = argparse.ArgumentParser(description="Preprocessing for Synthetic Human.")
parser.add_argument(
    "--data-dir", type=str, help="Directory that contains raw Synthetic Human data."
)
parser.add_argument(
    "--out-dir", type=str, help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--seqname", type=str, default="leonard", help="Sequence to process."
)
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")

if __name__ == "__main__":
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)
    out_dir = os.path.join(args.out_dir, seq_name)

    annots = np.load(os.path.join(data_dir, "annots.npy"), allow_pickle=True).item()
    # motion = dict(np.load(os.path.join(data_dir, "motion.npz"), allow_pickle=True))
    motion_dir = os.path.join(data_dir, "params")
    cameras = annots["cams"]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gender = "neutral"  # or "neutral"
    body_model_smpl = SMPL(model_path="./data/SMPLX/smpl", gender=gender).cuda()

    # cam_names = list(range(0, 10))
    # Use only one camera for now
    cam_names = [0]

    cam_names = ["{:02}".format(cam_name) for cam_name in cam_names]

    all_cam_params = {"all_cam_names": cam_names}

    shape = []
    global_orient = []
    body_pose = []
    transl = []
    vertices_smpl_space = []

    for cam_idx, cam_name in enumerate(cam_names):
        intrinsic = cameras["K"][cam_idx]
        # D = cameras["D"][cam_idx]
        R = cameras["R"][cam_idx]
        T = cameras["T"][cam_idx] / 1000.0
        # R is 3x3, T is 3x3, construct 4x4 extrinsic matrix
        extrinsic = np.block([[R, T], [0, 0, 0, 1]])

        img_in_dir = os.path.join(data_dir, "images/{}".format(cam_name))
        mask_in_dir = os.path.join(data_dir, "mask/{}".format(cam_name))

        img_out_dir = os.path.join(out_dir, "images/{}".format(cam_name))
        os.makedirs(img_out_dir, exist_ok=True)
        albedo_out_dir = os.path.join(out_dir, "albedos_png/{}".format(cam_name))
        os.makedirs(albedo_out_dir, exist_ok=True)
        normal_out_dir = os.path.join(out_dir, "normals_png/{}".format(cam_name))
        os.makedirs(normal_out_dir, exist_ok=True)
        mask_out_dir = os.path.join(out_dir, "masks/{}".format(cam_name))
        os.makedirs(mask_out_dir, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(img_in_dir, "*.jpg")))

        cam_params = {
            "intrinsic": intrinsic.tolist(),
            "extrinsic": extrinsic.tolist(),
            "distortion": [0, 0, 0, 0],
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
            idx = int(os.path.basename(img_file)[:-4])
            frame_index = idx

            mask_file = os.path.join(
                mask_in_dir, os.path.basename(img_file)[:-4] + ".png"
            )

            # We only process SMPL parameters in world coordinate
            if cam_idx == 0:
                params = np.load(
                    os.path.join(motion_dir, "{}.npy".format(idx)), allow_pickle=True
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

            # load mask_file and save as .npy
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            np.save(os.path.join(mask_out_dir, "mask_{:04d}.npy".format(idx)), mask)
            # load albedo file and save as .png (in linear RGB space)
            albedo_file = os.path.join(
                out_dir,
                "albedos/{}".format(cam_name),
                "albedo_0001_{:04d}.exr".format(idx),
            )
            albedo = pyexr.open(albedo_file).get()
            albedo = np.clip(albedo, 0, 1)
            # No need to convert to sRGB, since we will use the linear albedo
            # albedo = np.power(albedo, 1.0 / 2.2)
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(albedo_out_dir, "albedo_{:04d}.png".format(idx)),
                cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR),
            )
            # load normal file and save as .png (in OpenGL camera space)
            normal_file = os.path.join(
                out_dir,
                "normals/{}".format(cam_name),
                "normal_0001_{:04d}.exr".format(idx),
            )
            normal = pyexr.open(normal_file).get()
            normal = normal * 2 - 1 # convert to [-1, 1]
            normal = normal @ R.T # convert to OpenCV camera space
            normal = normal * np.array([1, -1, -1]) # convert to OpenGL camera space
            normal = (normal + 1) / 2 # convert to [0, 1]
            normal = (normal.clip(0, 1) * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(normal_out_dir, "normal_{:04d}.png".format(idx)),
                cv2.cvtColor(normal, cv2.COLOR_RGB2BGR),
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
