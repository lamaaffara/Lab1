"""Method 4 (docx Sec 4.1): Content-Preserving Warps style baseline.

Regular grid mesh (not a matched-point triangulation) solved with all three
CPW terms: data + local-rigidity (ASAP) + a homography-prior term pulling
every vertex toward a pre-fit global H. Per the docx's fairness substitution
(Sec 4, "Key implementation differences ... it is acceptable to substitute
the paper's own RANSAC-free inlier filter for correspondence input"), the
correspondences come from the shared relative-attribute filter rather than
RANSAC, so this differs from method_li_tang.py in exactly one respect: grid
mesh vs. Delaunay-over-matches mesh.
"""
import time

import numpy as np

from common import (GRID_NX, GRID_NY, MIN_INLIERS, W_ASAP, W_DATA, W_PRIOR, GridLocator,
                     MethodResult, apply_homography, barycentric_warp_fn, build_grid_mesh,
                     count_inverted_triangles, fit_global_homography_dlt, identity_result,
                     mesh_solve, relative_attribute_filter, render_affine_mesh)


def _aggregate_by_vertex(vertex_ids: np.ndarray, targets: np.ndarray):
    """Multiple matches can snap to the same nearest grid vertex; average
    their targets so the data term stays a well-posed one-constraint-per-vertex."""
    buckets: dict = {}
    for vid, t in zip(vertex_ids, targets):
        buckets.setdefault(int(vid), []).append(t)
    idx = np.array(list(buckets.keys()))
    tgt = np.array([np.mean(v, axis=0) for v in buckets.values()])
    return idx, tgt


def run_method(img1: np.ndarray, img2: np.ndarray, corr, render: bool = True) -> MethodResult:
    h, w = img1.shape[:2]
    t0 = time.perf_counter()

    inlier_mask = relative_attribute_filter(corr)
    if inlier_mask.sum() < MIN_INLIERS:
        return identity_result(img1, img2, time.perf_counter() - t0,
                                {"num_inliers": int(inlier_mask.sum())}, render)
    src, dst = corr.src_pts[inlier_mask], corr.dst_pts[inlier_mask]
    H_prior = fit_global_homography_dlt(src, dst)

    vertices, triangles, _ = build_grid_mesh(w, h, GRID_NX, GRID_NY)
    locator = GridLocator(w, h, GRID_NX, GRID_NY)
    prior_targets = apply_homography(H_prior, vertices)

    tri_ids = locator.locate(src)
    vpos = vertices[tri_ids]
    nearest_vertex = tri_ids[np.arange(len(src)), np.linalg.norm(vpos - src[:, None, :], axis=2).argmin(axis=1)]
    data_idx, data_targets = _aggregate_by_vertex(nearest_vertex, dst)

    solved = mesh_solve(vertices, triangles, data_idx, data_targets, W_DATA, W_ASAP,
                         prior_targets, W_PRIOR)
    elapsed = time.perf_counter() - t0

    warp_fn = barycentric_warp_fn(vertices, solved, locator.locate)
    n_inverted = count_inverted_triangles(vertices, solved, triangles)

    warped = coverage = None
    if render:
        warped, coverage = render_affine_mesh(img1, img2.shape, vertices, solved, triangles)

    return MethodResult(warped, coverage, warp_fn, elapsed, n_inverted,
                         {"num_inliers": int(inlier_mask.sum()), "H_prior": H_prior})
