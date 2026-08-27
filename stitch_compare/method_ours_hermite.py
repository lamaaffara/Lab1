"""Method 8 (docx Sec 1, best-performing configuration): identical mesh
construction and solve to method_ours_barycentric.py (Delaunay-over-matches,
data + ASAP, no global homography term), but rendered/queried with the
Hermite/Jacobian-enriched interpolation from docx Sec 1.6, which adds local
rotation/scale variation within a triangle that plain affine barycentric
interpolation cannot represent.
"""
import time

import numpy as np

from common import (MIN_INLIERS, SUBDIVIDE_K, W_ASAP, W_DATA, MethodResult,
                     compute_vertex_jacobians, count_inverted_triangles, build_delaunay_mesh,
                     hermite_warp_fn, identity_result, make_delaunay_locator, mesh_solve,
                     relative_attribute_filter, render_hermite_mesh)


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
    jacobians = compute_vertex_jacobians(len(vertices), data_idx,
                                          corr.src_scale[inlier_mask], corr.dst_scale[inlier_mask],
                                          corr.src_angle[inlier_mask], corr.dst_angle[inlier_mask])
    elapsed = time.perf_counter() - t0

    locate = make_delaunay_locator(delaunay, triangles)
    warp_fn = hermite_warp_fn(vertices, solved, jacobians, locate)
    n_inverted = count_inverted_triangles(vertices, solved, triangles)

    warped = coverage = None
    if render:
        warped, coverage = render_hermite_mesh(img1, img2.shape, vertices, triangles,
                                                 jacobians, solved, k=SUBDIVIDE_K)

    return MethodResult(warped, coverage, warp_fn, elapsed, n_inverted,
                         {"num_inliers": int(inlier_mask.sum())})
