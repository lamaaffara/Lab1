"""Method 7 (docx Sec 1, ablation A): our mesh construction and solve --
Delaunay-over-matches, data + local-rigidity (ASAP) terms, NO global
homography term at all -- rendered with plain barycentric interpolation
(no Hermite/Jacobian enrichment; see method_ours_hermite.py for that).

Differs from method_li_tang.py by exactly one term: w_prior = 0 here.
"""
import time

import numpy as np

from common import (MIN_INLIERS, W_ASAP, W_DATA, MethodResult, barycentric_warp_fn,
                     build_delaunay_mesh, count_inverted_triangles, identity_result,
                     make_delaunay_locator, mesh_solve, relative_attribute_filter,
                     render_affine_mesh)


def run_method(img1: np.ndarray, img2: np.ndarray, corr, render: bool = True) -> MethodResult:
    h, w = img1.shape[:2]
    t0 = time.perf_counter()

    inlier_mask = relative_attribute_filter(corr)
    if inlier_mask.sum() < MIN_INLIERS:
        return identity_result(img1, img2, time.perf_counter() - t0,
                                {"num_inliers": int(inlier_mask.sum())}, render)
    src, dst = corr.src_pts[inlier_mask], corr.dst_pts[inlier_mask]

    vertices, triangles, delaunay = build_delaunay_mesh(src, w, h)
    data_idx = np.arange(len(src))

    solved = mesh_solve(vertices, triangles, data_idx, dst, W_DATA, W_ASAP,
                         prior_targets=None, w_prior=0.0)
    elapsed = time.perf_counter() - t0

    locate = make_delaunay_locator(delaunay, triangles)
    warp_fn = barycentric_warp_fn(vertices, solved, locate)
    n_inverted = count_inverted_triangles(vertices, solved, triangles)

    warped = coverage = None
    if render:
        warped, coverage = render_affine_mesh(img1, img2.shape, vertices, solved, triangles)

    return MethodResult(warped, coverage, warp_fn, elapsed, n_inverted,
                         {"num_inliers": int(inlier_mask.sum())})
