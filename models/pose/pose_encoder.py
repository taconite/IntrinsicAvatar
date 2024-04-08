import numpy as np
import torch
import torch.nn as nn

import models


@models.register("dummy_pose_encoder")
class DummyPoseEncoder(nn.Module):
    """Dummy pose encoder."""
    def __init__(self, config):
        super().__init__()

    def forward(self, rots, Jtrs):
        return torch.empty(rots.shape[0], 0, device=rots.device)


@models.register("leap")
class HierarchicalPoseEncoder(nn.Module):
    """Hierarchical encoder from LEAP."""

    def __init__(self, config):
        super().__init__()

        self.num_joints = config.get("num_joints", 24)
        self.rel_joints = config.get("rel_joints", False)
        dim_per_joint = config.get("dim_per_joint", 6)
        out_dim = config.get("out_dim", -1)
        self.ktree_parents = np.array(
            [
                -1,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                9,
                9,
                12,
                13,
                14,
                16,
                17,
                18,
                19,
                20,
                21,
            ],
            dtype=np.int32,
        )

        self.layer_0 = nn.Linear(
            9 * self.num_joints + 3 * self.num_joints, dim_per_joint
        )
        dim_feat = 13 + dim_per_joint

        layers = []
        for idx in range(self.num_joints):
            layer = nn.Sequential(
                nn.Linear(dim_feat, dim_feat),
                nn.ReLU(),
                nn.Linear(dim_feat, dim_per_joint),
            )

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer = nn.Identity()
            self.n_output_dims = self.num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(self.num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(self, rots, Jtrs):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = (
                    Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                )
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat(
            [rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1
        )
        global_feat = self.layer_0(global_feat)

        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(
                    Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :],
                    dim=-1,
                    keepdim=True,
                )
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        out = self.out_layer(out)
        return out
