"""Method 6 (docx Sec 4.3): matched-point Delaunay mesh with an active
homography-prior term (Li et al. 2019 / Tang et al. 2023 style).

Structurally this is "our mesh construction (Sec 1.4), but with the
homography-prior term left active" -- it differs from method_ours_barycentric.py
by exactly one term, which is the point of this baseline: it is the direct,
controlled test of whether removing the prior (our method) costs accuracy.
Per the docx's fairness substitution, correspondences come from the shared
relative-attribute filter rather than a RANSAC-conditioned set upstream.
"""
import time

import numpy as np

from common import (MIN_INLIERS, W_ASAP, W_DATA, W_PRIOR, MethodResult, apply_homography,
                     barycentric_warp_fn, build_delaunay_mesh, count_inverted_triangles,
                     fit_global_homography_dlt, identity_result, make_delaunay_locator,
                     mesh_solve, relative_attribute_filter, render_affine_mesh)


def run_method(img1: np.ndarray, img2: np.ndarray, corr, render: bool = True) -> MethodResult:
    h, w = img1.shape[:2]
    t0 = time.perf_counter()

    inlier_mask = relative_attribute_filter(corr)
    if inlier_mask.sum() < MIN_INLIERS:
        return identity_result(img1, img2, time.perf_counter() - t0,
                                {"num_inliers": int(inlier_mask.sum())}, render)
    src, dst = corr.src_pts[inlier_mask], corr.dst_pts[inlier_mask]
    H_prior = fit_global_homography_dlt(src, dst)

    vertices, triangles, delaunay = build_delaunay_mesh(src, w, h)
    n_inliers = len(src)
    data_idx = np.arange(n_inliers)
    prior_targets = apply_homography(H_prior, vertices)

    solved = mesh_solve(vertices, triangles, data_idx, dst, W_DATA, W_ASAP,
                         prior_targets, W_PRIOR)
    elapsed = time.perf_counter() - t0

    locate = make_delaunay_locator(delaunay, triangles)
    warp_fn = barycentric_warp_fn(vertices, solved, locate)
    n_inverted = count_inverted_triangles(vertices, solved, triangles)

    warped = coverage = None
    if render:
        warped, coverage = render_affine_mesh(img1, img2.shape, vertices, solved, triangles)

    return MethodResult(warped, coverage, warp_fn, elapsed, n_inverted,
                         {"num_inliers": int(inlier_mask.sum()), "H_prior": H_prior})
