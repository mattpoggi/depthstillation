#
# Classes and functions in this script are taken from https://github.com/vt-vl-lab/3d-photo-inpainting/blob/master/bilateral_filtering.py
# Licensed under MIT license
#

import numpy as np
from functools import reduce
import time

__all__ = ["sparse_bilateral_filtering"]


def sparse_bilateral_filtering(
    depth,
    image,
    filter_size,
    sigma_r=0.5,
    sigma_s=4.0,
    depth_threshold=0.04,
    HR=False,
    mask=None,
    gsHR=True,
    edge_id=None,
    num_iter=None,
    num_gs_iter=None,
):

    save_discontinuities = []
    vis_depth = depth.copy()
    backup_vis_depth = vis_depth.copy()

    depth_max = vis_depth.max()
    depth_min = vis_depth.min()
    vis_image = image.copy()
    for i in range(num_iter):
        vis_image = image.copy()
        u_over, b_over, l_over, r_over = vis_depth_discontinuity(
            vis_depth, depth_threshold, mask=mask
        )
        vis_image[u_over > 0] = np.array([0, 0, 0])
        vis_image[b_over > 0] = np.array([0, 0, 0])
        vis_image[l_over > 0] = np.array([0, 0, 0])
        vis_image[r_over > 0] = np.array([0, 0, 0])

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        discontinuity_map[depth == 0] = 1
        save_discontinuities.append(discontinuity_map)
        if mask is not None:
            discontinuity_map[mask == 0] = 0
        vis_depth = bilateral_filter(
            vis_depth,
            sigma_r=sigma_r,
            sigma_s=sigma_s,
            discontinuity_map=discontinuity_map,
            HR=HR,
            mask=mask,
            window_size=filter_size[i],
        )

    return vis_depth


def vis_depth_discontinuity(
    depth, depth_threshold, vis_diff=False, label=False, mask=None
):
    if label == False:
        disp = 1.0 / depth
        u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > depth_threshold).astype(np.float32)
        b_over = (np.abs(b_diff) > depth_threshold).astype(np.float32)
        l_over = (np.abs(l_diff) > depth_threshold).astype(np.float32)
        r_over = (np.abs(r_diff) > depth_threshold).astype(np.float32)
    else:
        disp = depth
        u_diff = (disp[1:, :] * disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] * disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] * disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] * disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > 0).astype(np.float32)
        b_over = (np.abs(b_diff) > 0).astype(np.float32)
        l_over = (np.abs(l_diff) > 0).astype(np.float32)
        r_over = (np.abs(r_diff) > 0).astype(np.float32)
    u_over = np.pad(u_over, 1, mode="constant")
    b_over = np.pad(b_over, 1, mode="constant")
    l_over = np.pad(l_over, 1, mode="constant")
    r_over = np.pad(r_over, 1, mode="constant")
    u_diff = np.pad(u_diff, 1, mode="constant")
    b_diff = np.pad(b_diff, 1, mode="constant")
    l_diff = np.pad(l_diff, 1, mode="constant")
    r_diff = np.pad(r_diff, 1, mode="constant")

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]


def bilateral_filter(
    depth,
    sigma_s,
    sigma_r,
    window_size,
    discontinuity_map=None,
    HR=False,
    mask=None,
):
    sort_time = 0
    replace_time = 0
    filter_time = 0
    init_time = 0
    filtering_time = 0

    midpt = window_size // 2
    ax = np.arange(-midpt, midpt + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    if discontinuity_map is not None:
        spatial_term = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_s ** 2))

    # padding
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1, 1), (1, 1)), "edge")
    pad_depth = np.pad(depth, (midpt, midpt), "edge")
    if discontinuity_map is not None:
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1, 1), (1, 1)), "edge")
        pad_discontinuity_map = np.pad(discontinuity_map, (midpt, midpt), "edge")
        pad_discontinuity_hole = 1 - pad_discontinuity_map
    # filtering
    output = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1, 1])
    if discontinuity_map is not None:
        pad_discontinuity_patches = rolling_window(
            pad_discontinuity_map, [window_size, window_size], [1, 1]
        )
        pad_discontinuity_hole_patches = rolling_window(
            pad_discontinuity_hole, [window_size, window_size], [1, 1]
        )

    if mask is not None:
        pad_mask = np.pad(mask, (midpt, midpt), "constant")
        pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1, 1])
    from itertools import product

    if discontinuity_map is not None:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if mask is not None and mask[pi, pj] == 0:
                    continue
                if discontinuity_map is not None:
                    if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size // 2, window_size // 2]
                if discontinuity_map is not None:
                    coef = discontinuity_holes.astype(np.float32)
                    if mask is not None:
                        coef = coef * pad_mask_patches[pi, pj]
                else:
                    range_term = np.exp(
                        -((depth_patch - patch_midpt) ** 2) / (2.0 * sigma_r ** 2)
                    )
                    coef = spatial_term * range_term
                if coef.max() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.max() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef / (coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]
    else:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if discontinuity_map is not None:
                    if (
                        pad_discontinuity_patches[pi, pj][
                            window_size // 2, window_size // 2
                        ]
                        == 1
                    ):
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = 1.0 - discontinuity_patch
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size // 2, window_size // 2]
                range_term = np.exp(
                    -((depth_patch - patch_midpt) ** 2) / (2.0 * sigma_r ** 2)
                )
                if discontinuity_map is not None:
                    coef = spatial_term * range_term * discontinuity_holes
                else:
                    coef = spatial_term * range_term
                if coef.sum() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.sum() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef / (coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]

    return output


def rolling_window(a, window, strides):
    assert (
        len(a.shape) == len(window) == len(strides)
    ), "'a', 'window', 'strides' dimension mismatch"
    shape_fn = lambda i, w, s: (a.shape[i] - w) // s + 1
    shape = [shape_fn(i, w, s) for i, (w, s) in enumerate(zip(window, strides))] + list(
        window
    )

    def acc_shape(i):
        if i + 1 >= len(a.shape):
            return 1
        else:
            return reduce(lambda x, y: x * y, a.shape[i + 1 :])

    _strides = [acc_shape(i) * s * a.itemsize for i, s in enumerate(strides)] + list(
        a.strides
    )

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)
