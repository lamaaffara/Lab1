"""Method 5 (docx Sec 4.2): APAP / Moving-DLT.

Not a triangulated mesh -- a regular grid of cells, each with its OWN locally
fit homography via a weighted DLT (every correspondence contributes to every
cell, weighted by a Gaussian-like spatial kernel centered on that cell).
Per the docx's fairness substitution, correspondences come from the shared
relative-attribute filter instead of a RANSAC-conditioned set upstream.

Point queries and rendering both use plain nearest-cell assignment (no extra
inter-cell blending) -- the docx lists boundary blending as an optional
smoothing step, not a required part of the method.
"""
import time

import numpy as np

from common import (GRID_NX, GRID_NY, MIN_INLIERS, MethodResult, apply_homography,
                     identity_result, relative_attribute_filter, render_via_triangles)

SIGMA_FRAC = 1.0 / GRID_NX   # kernel width as a fraction of image width
W_MIN = 1e-2


def _weighted_dlt(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Same normalized-DLT construction as the base-paper solve, but every
    correspondence's two rows are scaled by sqrt(weight) before the SVD."""
    centroid_s, centroid_d = src.mean(0), dst.mean(0)
    scale_s = np.sqrt(2) / max(np.linalg.norm(src - centroid_s, axis=1).mean(), 1e-8)
    scale_d = np.sqrt(2) / max(np.linalg.norm(dst - centroid_d, axis=1).mean(), 1e-8)
    T_s = np.array([[scale_s, 0, -scale_s * centroid_s[0]],
                    [0, scale_s, -scale_s * centroid_s[1]], [0, 0, 1]])
    T_d = np.array([[scale_d, 0, -scale_d * centroid_d[0]],
                    [0, scale_d, -scale_d * centroid_d[1]], [0, 0, 1]])
    src_n = (np.hstack([src, np.ones((len(src), 1))]) @ T_s.T)[:, :2]
    dst_n = (np.hstack([dst, np.ones((len(dst), 1))]) @ T_d.T)[:, :2]

    n = len(src_n)
    A = np.zeros((2 * n, 9))
    sw = np.sqrt(np.maximum(weights, 0))
    for i in range(n):
        x, y = src_n[i]
        xp, yp = dst_n[i]
        A[2 * i] = sw[i] * np.array([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A[2 * i + 1] = sw[i] * np.array([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_d) @ H_n @ T_s
    return H / H[2, 2]


def _cell_index(pts, w, h, nx, ny):
    cx = np.clip((pts[:, 0] / w * nx).astype(int), 0, nx - 1)
    cy = np.clip((pts[:, 1] / h * ny).astype(int), 0, ny - 1)
    return cx, cy


def run_method(img1: np.ndarray, img2: np.ndarray, corr, render: bool = True) -> MethodResult:
    h, w = img1.shape[:2]
    nx, ny = GRID_NX, GRID_NY
    sigma = SIGMA_FRAC * w

    t0 = time.perf_counter()
    inlier_mask = relative_attribute_filter(corr)
    if inlier_mask.sum() < MIN_INLIERS:
        return identity_result(img1, img2, time.perf_counter() - t0,
                                {"num_inliers": int(inlier_mask.sum())}, render)
    src, dst = corr.src_pts[inlier_mask], corr.dst_pts[inlier_mask]

    cell_w, cell_h = w / nx, h / ny
    centers = np.array([[(i + 0.5) * cell_w, (j + 0.5) * cell_h]
                         for j in range(ny) for i in range(nx)])
    homographies = np.zeros((ny * nx, 3, 3))
    for c_idx, center in enumerate(centers):
        d2 = ((src - center) ** 2).sum(axis=1)
        weights = np.maximum(np.exp(-d2 / (sigma ** 2)), W_MIN)
        homographies[c_idx] = _weighted_dlt(src, dst, weights)
    elapsed = time.perf_counter() - t0

    def warp_fn(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        cx, cy = _cell_index(pts, w, h, nx, ny)
        cell_id = cy * nx + cx
        out = np.zeros_like(pts)
        for cid in np.unique(cell_id):
            sel = cell_id == cid
            out[sel] = apply_homography(homographies[cid], pts[sel])
        return out

    warped = coverage = None
    if render:
        src_tris, dst_tris = [], []
        for j in range(ny):
            for i in range(nx):
                x0, x1 = i * cell_w, min((i + 1) * cell_w, w - 1)
                y0, y1 = j * cell_h, min((j + 1) * cell_h, h - 1)
                corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                H_cell = homographies[j * nx + i]
                mapped = apply_homography(H_cell, corners)
                src_tris += [corners[[0, 1, 2]], corners[[0, 2, 3]]]
                dst_tris += [mapped[[0, 1, 2]], mapped[[0, 2, 3]]]
        warped, coverage = render_via_triangles(img1, img2.shape, np.array(src_tris), np.array(dst_tris))

    return MethodResult(warped, coverage, warp_fn, elapsed, None,
                         {"num_inliers": int(inlier_mask.sum()), "grid": (nx, ny)})
