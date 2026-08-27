"""Method 1 (docx Sec 3.1): standard 4-point RANSAC + global homography.

cv2.findHomography(..., cv2.RANSAC) does its own outlier removal internally
(no relative-attribute filter here -- this is the classical baseline).
"""
import time

import cv2
import numpy as np

from common import MethodResult, warp_fn_from_homography

REPROJ_THRESH = 3.0


def run_method(img1: np.ndarray, img2: np.ndarray, corr, render: bool = True) -> MethodResult:
    t0 = time.perf_counter()
    H, mask = cv2.findHomography(corr.src_pts, corr.dst_pts, cv2.RANSAC, REPROJ_THRESH)
    if H is None:
        H = np.eye(3)
        mask = np.zeros((len(corr.src_pts), 1), dtype=np.uint8)
    elapsed = time.perf_counter() - t0

    warp_fn = warp_fn_from_homography(H)
    warped = coverage = None
    if render:
        out_size = (img2.shape[1], img2.shape[0])
        warped = cv2.warpPerspective(img1, H, out_size)
        coverage = cv2.warpPerspective(np.ones(img1.shape[:2], dtype=np.uint8), H, out_size) > 0

    return MethodResult(warped, coverage, warp_fn, elapsed, None,
                         {"num_inliers": int(mask.sum()), "H": H})
