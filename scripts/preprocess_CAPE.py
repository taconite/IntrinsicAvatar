import os
import glob
import torch

import cv2
import argparse

import numpy as np

from scipy.spatial.transform import Rotation as Rotation

# body models
from models.deformers.smplx import SMPL

from utils.smpl_renderer import Renderer

parser = argparse.ArgumentParser(description="Preprocessing for CAPE for animation.")
parser.add_argument(
    "--data-dir",
    type=str,
    default="/home/sfwang/Datasets/CAPE",
    help="Directory that contains raw CAPE data.",
)
parser.add_argument(
    "--out-dir", type=str,
    default="./load/animation",
    help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--src-subj-name",
    type=str,
    default="male-3-casual",
    help="Subject from which to load body shape.",
)
parser.add_argument(
    "--tgt-subj-name",
    type=str,
    default="00032",
    help="Subject from which to load motion.",
)
parser.add_argument(
    "--seq-name",
    type=str,
    default="shortlong_soccer",
    help="Sequence from which to load motion.",
)
parser.add_argument("--start", type=int, default=0, help="Start frame.")
parser.add_argument("--end", type=int, default=-1, help="End frame.")
parser.add_argument("--skip", type=int, default=1, help="Skip every n frames.")
parser.add_argument(
    "--rotate", action="store_true", help="Rotate 360 degrees of the last frame."
)
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")

if __name__ == "__main__":
    args = parser.parse_args()
    src_subj_name = args.src_subj_name
    tgt_subj_name = args.tgt_subj_name
    seq_name = args.seq_name
    out_dir = os.path.join(
        args.out_dir, src_subj_name, "cape_" + tgt_subj_name + "_" + seq_name
    )
    data_dir = os.path.join(args.data_dir, tgt_subj_name, seq_name)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")

    if args.visualize:
        vis_dir = os.path.join(out_dir, "smpl_vis", "images")
        os.makedirs(vis_dir, exist_ok=True)

    # vertices_out_dir = os.path.join(out_dir, "vertices")
    # os.makedirs(vertices_out_dir, exist_ok=True)

    # Load SMPL shape
    shape_file = f"./load/peoplesnapshot/{args.src_subj_name}/poses/anim_nerf_train.npz"
    betas = dict(np.load(shape_file))['betas']

    # Standard SMPL model
    gender = "male" if args.src_subj_name.startswith("male") else "female"
    print(gender)

    body_model_smpl = SMPL(model_path="./data/SMPLX/smpl", gender=gender).cuda()

    # Load CAPE poses
    # # smpl_params = dict(np.load("load/animation/male-3-casual/poses.npz"))
    # with open(os.path.join(args.data_dir, f"{seq_name}.pkl"), "rb") as f:
    #     data = pkl.load(f)
    # smpl_params = {"poses": data["smpl_poses"],
    #                "trans": data["smpl_trans"] / 100.0}

    # smpl_params["poses"][..., :3] = Rotation.from_matrix(root_orient).as_rotvec()
    pose_filepaths = sorted(
        glob.glob(os.path.join(data_dir, f"{seq_name}*.npz"))
    )

    smpl_params = {"poses": [], "trans": []}
    for pose_filepath in pose_filepaths:
        smpl_param = dict(np.load(pose_filepath))
        smpl_params["poses"].append(smpl_param["pose"])
        smpl_params["trans"].append(smpl_param["transl"])

    smpl_params = {
        "poses": np.stack(smpl_params["poses"], axis=0),
        "trans": np.stack(smpl_params["trans"], axis=0),
    }

    # Rotate root orientation around x-axis by 180 degrees
    root_orient = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix() @ (
        Rotation.from_rotvec(smpl_params["poses"][..., :3]).as_matrix()
    )
    smpl_params["poses"][..., :3] = Rotation.from_matrix(root_orient).as_rotvec()

    # Finally, set hand and feet poses to zero
    smpl_params["poses"][..., 21:24] = 0
    smpl_params["poses"][..., 24:27] = 0
    smpl_params["poses"][..., 30:33] = 0
    smpl_params["poses"][..., 33:36] = 0
    smpl_params["poses"][..., 60:] = 0

    # Load camera parameters
    camera = dict(np.load("load/animation/aist/cameras.npz"))

    height = camera["height"]
    width = camera["width"]

    K = np.eye(3)
    K[0, 0] = K[1, 1] = 2000
    K[0, 2] = height // 2
    K[1, 2] = width // 2
    intrinsic = K.copy()

    extrinsic = camera["extrinsic"].copy()

    thetas = smpl_params["poses"][..., :72]
    # transl = smpl_params["trans"] - smpl_params["trans"][0:1]
    # transl += (0, 0.15, 5)
    transl = smpl_params["trans"]

    if args.end == -1:
        args.end = len(thetas)
    else:
        args.end += 1
    global_orients = thetas[args.start:args.end:args.skip, :3].astype(np.float32)
    transls = transl[args.start:args.end:args.skip].astype(np.float32)
    body_poses = thetas[args.start:args.end:args.skip, 3:].astype(np.float32)

    extrinsics = [extrinsic.copy() for _ in range(len(global_orients))]
    trajectory = []
    if args.rotate:
        azimuths = np.linspace(0, 2 * np.pi, 50)
        transl_last = transl[-1].reshape([3, 1])
        for azimuth in azimuths:
            # Create rotation matrix around y-axis (azimuth)
            R_y = np.array([[np.cos(azimuth), 0, np.sin(azimuth)],
                            [0, 1, 0],
                            [-np.sin(azimuth), 0, np.cos(azimuth)]])

            # Combine rotation with translation
            R = R_y @ extrinsic[:3, :3]
            t = -R @ transl_last + transl_last + extrinsic[:3, 3:]

            # Store the extrinsic matrix (4x4 matrix in OpenCV format)
            extrinsic_matrix = np.block([[R, t], [0, 0, 0, 1]])
            trajectory.append(extrinsic_matrix)

            global_orients = np.concatenate(
                [global_orients, global_orients[-1][None].copy()], axis=0
            )
            transls = np.concatenate([transls, transls[-1][None].copy()], axis=0)
            body_poses = np.concatenate([body_poses, body_poses[-1][None].copy()], axis=0)

    extrinsics.extend(trajectory)
    intrinsics = [intrinsic.copy() for _ in range(len(global_orients))]
    heights = [height for _ in range(len(global_orients))]
    widths = [width for _ in range(len(global_orients))]

    shape = betas.copy()

    for idx, (global_orient, body_pose, transl) in enumerate(
        zip(global_orients, body_poses, transls)
    ):
        print("Processing: {}".format(idx))

        # base_name = os.path.basename(img_file).split(".")[0]
        R = extrinsics[idx][:3, :3].copy()
        T = extrinsics[idx][:3, 3:].copy()

        poses_torch = torch.from_numpy(body_pose[None]).cuda()
        betas_torch = torch.from_numpy(betas).cuda()
        global_orient_torch = torch.from_numpy(global_orient[None]).cuda()
        transl_torch = torch.from_numpy(transl[None]).cuda()

        # SMPL mesh via standard SMPL
        smpl_outputs = body_model_smpl(
            betas=betas_torch,
            body_pose=poses_torch,
            global_orient=global_orient_torch,
            transl=transl_torch,
        )
        verts_smpl = smpl_outputs.vertices.detach().cpu().numpy()[0]

        # Visualize SMPL mesh
        if args.visualize:
            assert verts_smpl.shape == (6890, 3)

            # # Save SMPL vertices
            # np.save(
            #     os.path.join(vertices_out_dir, "verts_{:04d}.npy".format(idx)),
            #     verts_smpl,
            # )

            renderer = Renderer(
                height=height,
                width=width,
                faces=body_model_smpl.faces,
            )

            render_cameras = {
                "K": [intrinsics[idx]],
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

            images = [np.zeros((height, width, 3), dtype=np.uint8)]
            smpl_image = renderer.render(
                render_data,
                render_cameras,
                images,
                use_white=False,
                add_back=True,
            )[0]

            cv2.imwrite(
                os.path.join(vis_dir, f"{idx:04d}.png"),
                cv2.cvtColor(smpl_image, cv2.COLOR_RGB2BGR),
            )

            del renderer

    # Save SMPL parameters
    out_filename = os.path.join(out_dir, "poses.npz")
    np.savez(
        out_filename,
        poses=np.concatenate([global_orients, body_poses], axis=-1),
        trans=transls,
    )
    # Save new camera parameters
    out_filename = os.path.join(out_dir, "cameras.npz")

    np.savez(
        out_filename,
        height=heights,
        width=widths,
        extrinsic=extrinsics,
        intrinsic=intrinsics,
    )
