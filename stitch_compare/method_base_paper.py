"""Method 3 (docx Sec 2): the base paper's own pipeline being replaced --
relative-attribute histogram filter (no RANSAC) + global homography via the
DLT-eigen solve. Isolates the effect of Stage 4/5 of our method, since
Stages 1-2 (detection, filtering) are identical to every other method here.
"""
import time

import cv2
import numpy as np

from common import (MIN_INLIERS, MethodResult, fit_global_homography_dlt, identity_result,
                     relative_attribute_filter, warp_fn_from_homography)


def run_method(img1: np.ndarray, img2: np.ndarray, corr, render: bool = True) -> MethodResult:
    t0 = time.perf_counter()
    inlier_mask = relative_attribute_filter(corr)
    if inlier_mask.sum() < MIN_INLIERS:
        return identity_result(img1, img2, time.perf_counter() - t0,
                                {"num_inliers": int(inlier_mask.sum())}, render)
    H = fit_global_homography_dlt(corr.src_pts[inlier_mask], corr.dst_pts[inlier_mask])
    elapsed = time.perf_counter() - t0

    warp_fn = warp_fn_from_homography(H)
    warped = coverage = None
    if render:
        out_size = (img2.shape[1], img2.shape[0])
        warped = cv2.warpPerspective(img1, H, out_size)
        coverage = cv2.warpPerspective(np.ones(img1.shape[:2], dtype=np.uint8), H, out_size) > 0

    return MethodResult(warped, coverage, warp_fn, elapsed, None,
                         {"num_inliers": int(inlier_mask.sum()), "H": H})
