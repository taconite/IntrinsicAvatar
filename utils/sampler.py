import numpy as np
import cv2


""" Copied and modified from InstantAvatar, https://github.com/tijiang13/InstantAvatar
"""


class EdgeSampler:
    def __init__(self,
                 num_sample,
                 ratio_mask=0.6,
                 ratio_edge=0.3,
                 kernel_size=32):

        assert ratio_mask >= 0.0
        assert ratio_edge >= 0.0
        assert ratio_edge + ratio_mask <= 1.0

        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self.num_mask = int(num_sample * ratio_mask)
        self.num_edge = int(num_sample * ratio_edge)
        self.num_rand = num_sample - self.num_mask - self.num_edge

    def sample(self, mask, *args):
        mask = mask.reshape(-1)

        # calculate the edge area
        mask_i = cv2.erode(mask, self.kernel)
        mask_o = cv2.dilate(mask, self.kernel)
        mask_e = mask_o - mask_i

        mask_loc, *_ = np.where(mask)
        edge_loc, *_ = np.where(mask_e)

        mask_idx = np.random.randint(0, len(mask_loc), self.num_mask)
        edge_idx = np.random.randint(0, len(edge_loc), self.num_edge)
        rand_idx = np.random.randint(0, len(mask), self.num_rand)

        mask_idx = mask_loc[mask_idx]
        edge_idx = edge_loc[edge_idx]

        indices = np.concatenate([mask_idx, edge_idx, rand_idx], axis=0)
        output = [mask[indices]]
        for d in args:
            d = d.reshape(len(mask), -1)
            output.append(d[indices])
        return output


class UniformSampler:
    def __init__(self,
                 num_sample):

        self.num_rand = num_sample

    def sample(self, mask, *args):
        mask = mask.reshape(-1)

        indices = np.random.randint(0, len(mask), self.num_rand)

        output = [mask[indices]]
        for d in args:
            d = d.reshape(len(mask), -1)
            output.append(d[indices])
        return output


class BalancedSampler:
    def __init__(self, num_sample):
        self.num_fg_sample = num_sample // 2
        self.num_bg_sample = num_sample - self.num_fg_sample

    def sample(self, mask, *args):
        mask = mask.reshape(-1)

        # indices = np.random.randint(0, len(mask), self.num_rand)
        fg_indices = np.where(mask == 1)[0]
        sampled_fg_indices = np.random.choice(
            fg_indices,
            self.num_fg_sample,
            replace=False if len(fg_indices) > self.num_fg_sample else True,
        )
        if hasattr(self, 'aabb_mask'):
            bg_indices = np.where((self.aabb_mask) & (mask == 0))[0]
        else:
            bg_indices = np.where(mask == 0)[0]
        sampled_bg_indices = np.random.choice(
            bg_indices,
            self.num_bg_sample,
            replace=False if len(bg_indices) > self.num_bg_sample else True,
        )
        indices = np.concatenate([sampled_fg_indices, sampled_bg_indices], axis=0)

        output = [mask[indices]]
        for d in args:
            d = d.reshape(len(mask), -1)
            output.append(d[indices])
        return output


class PatchSampler():
    def __init__(self, num_patch=4, patch_size=20, ratio_mask=0.9, dilate=0):
        self.n = num_patch
        self.patch_size = patch_size
        self.p = ratio_mask
        self.dilate = dilate
        assert self.patch_size % 2 == 0, "patch size has to be even"

    def sample(self, mask, *args):
        patch = (self.patch_size, self.patch_size)
        shape = mask.shape[:2]

        if np.random.rand() < self.p:
            o = patch[0] // 2
            if self.dilate > 0:
                m = cv2.dilate(mask, np.ones((self.dilate, self.dilate))) > 0
            else:
                m = mask
            valid = m[o:-o, o:-o] > 0
            (xs, ys) = np.where(valid)
            idx = np.random.choice(len(xs), size=self.n, replace=False)
            x, y = xs[idx], ys[idx]
        else:
            x = np.random.randint(0, shape[0] - patch[0], size=self.n)
            y = np.random.randint(0, shape[1] - patch[1], size=self.n)
        output = []
        for d in [mask, *args]:
            patches = []
            for xi, yi in zip(x, y):
                p = d[xi:xi + patch[0], yi:yi + patch[1]]
                patches.append(p)
            patches = np.stack(patches, axis=0)
            if patches.shape[-1] == 1: patches = patches.squeeze(-1)
            output.append(patches)
        return output
