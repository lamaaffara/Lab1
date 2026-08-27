"""Shared infrastructure used by every method_*.py file.

Covers: SIFT detect/match (docx Sec 1.2), the relative-attribute inlier
filter (docx Sec 1.3), global-homography DLT, Delaunay/grid mesh
construction, the data+ASAP(+prior) sparse mesh solve (docx Sec 1.4-1.5,
Sec 4), barycentric and Hermite/Jacobian interpolation (docx Sec 1.6), and
triangle-based rendering (docx Sec 1.7). Every method_*.py imports from
here instead of duplicating this math.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# Shared constants (kept identical across methods for a fair comparison)
# ---------------------------------------------------------------------------
RATIO_TEST = 0.75          # Lowe ratio test threshold for BF matching
HIST_BINS = 20             # histogram bins for the relative-attribute filter (tuned:
                           # 50 bins made individual bins too narrow to ever clear the
                           # 0.1 probability-mass threshold on some seeds, rejecting
                           # every match; 20 keeps bins coarse enough to be robust
                           # while still discriminating the dominant inlier peak)
DXY_THRESH = 0.1           # probability-mass threshold for dx/dy bins (docx Sec 1.3)
GRID_NX = 8                # grid resolution shared by CPW and APAP (fairness: same density)
GRID_NY = 8
SUBDIVIDE_K = 4            # sub-triangle subdivision factor for Hermite rendering
W_DATA = 1.0
W_ASAP = 1.0
W_PRIOR = 0.35             # docx Sec 4.3: reference weight for homography-prior baselines
APAP_SIGMA_FRAC = 1.0 / GRID_NX   # APAP Gaussian kernel width, relative to image width
APAP_W_MIN = 1e-2


@dataclass
class MethodResult:
    """Common return type for every method_*.py `run_method`."""
    warped: Optional[np.ndarray]        # rendered warp of img1 into img2's frame, or None
    coverage_mask: Optional[np.ndarray] # bool mask, True where `warped` has valid content
    warp_fn: Callable[[np.ndarray], np.ndarray]  # (N,2) image-1 pts -> (N,2) predicted image-2 pts
    elapsed: float                      # seconds spent in outlier-removal + warp-estimation only
    num_inverted_triangles: Optional[int] = None  # None if the method is not mesh-based
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1-2: SIFT detect/match + relative-attribute inlier filter
# ---------------------------------------------------------------------------

@dataclass
class Correspondences:
    """Raw matched SIFT correspondences shared verbatim by every method."""
    src_pts: np.ndarray   # (M,2) positions in image 1
    dst_pts: np.ndarray   # (M,2) positions in image 2
    src_scale: np.ndarray  # (M,) kp.size in image 1
    dst_scale: np.ndarray
    src_angle: np.ndarray  # (M,) kp.angle in RADIANS, image 1
    dst_angle: np.ndarray


def sift_detect_and_match(img1: np.ndarray, img2: np.ndarray) -> Correspondences:
    """SIFT detect + BF nearest-neighbor match + ratio test (docx Sec 1.2)."""
    sift = cv2.SIFT_create()
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < RATIO_TEST * n.distance]

    src_pts = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64)
    src_scale = np.array([kp1[m.queryIdx].size for m in good], dtype=np.float64)
    dst_scale = np.array([kp2[m.trainIdx].size for m in good], dtype=np.float64)
    src_angle = np.deg2rad(np.array([kp1[m.queryIdx].angle for m in good], dtype=np.float64))
    dst_angle = np.deg2rad(np.array([kp2[m.trainIdx].angle for m in good], dtype=np.float64))
    return Correspondences(src_pts, dst_pts, src_scale, dst_scale, src_angle, dst_angle)


def _hist_mask_threshold(values: np.ndarray, bins: int, thresh: float) -> np.ndarray:
    hist, edges = np.histogram(values, bins=bins)
    prob = hist / max(hist.sum(), 1)
    qualifying = np.where(prob > thresh)[0]
    bin_idx = np.clip(np.digitize(values, edges[1:-1]), 0, bins - 1)
    return np.isin(bin_idx, qualifying)


def _hist_mask_mode(values: np.ndarray, bins: int) -> np.ndarray:
    hist, edges = np.histogram(values, bins=bins)
    top = int(np.argmax(hist))
    bin_idx = np.clip(np.digitize(values, edges[1:-1]), 0, bins - 1)
    return bin_idx == top


def train_holdout_split(corr: Correspondences, holdout_frac: float, rng: np.random.Generator):
    """Split the RAW match set (before any method's own outlier removal) into
    a 'used' set fed identically to every method, and a held-out set whose
    source positions serve as the docx Sec 6.2 'held-out matches' query set.
    """
    m = len(corr.src_pts)
    perm = rng.permutation(m)
    n_hold = max(1, int(round(m * holdout_frac)))
    hold_idx, used_idx = perm[:n_hold], perm[n_hold:]

    used = Correspondences(
        corr.src_pts[used_idx], corr.dst_pts[used_idx],
        corr.src_scale[used_idx], corr.dst_scale[used_idx],
        corr.src_angle[used_idx], corr.dst_angle[used_idx],
    )
    holdout_src = corr.src_pts[hold_idx]
    return used, holdout_src


MIN_INLIERS = 4  # below this, a homography/mesh solve is numerically singular or meaningless


def identity_result(img1: np.ndarray, img2: np.ndarray, elapsed: float, meta: dict,
                     render: bool) -> "MethodResult":
    """Degenerate-case fallback (near-zero inliers survived filtering) so a
    pathological seed reports as a clearly-flagged identity warp instead of
    an arbitrary numerical artifact from an under-determined solve."""
    meta = dict(meta)
    meta["degenerate"] = True
    warp_fn = lambda pts: np.array(pts, dtype=np.float64, copy=True)
    warped = coverage = None
    if render:
        h, w = img2.shape[:2]
        warped = cv2.resize(img1, (w, h)) if img1.shape[:2] != (h, w) else img1.copy()
        coverage = np.ones((h, w), dtype=bool)
    return MethodResult(warped, coverage, warp_fn, elapsed, None, meta)


def relative_attribute_filter(corr: Correspondences) -> np.ndarray:
    """docx Sec 1.3: histogram-based relative-attribute inlier filter.

    Returns a boolean inlier mask of length M (no RANSAC, O(M), single pass).
    """
    dx = corr.src_pts[:, 0] - corr.dst_pts[:, 0]
    dy = corr.src_pts[:, 1] - corr.dst_pts[:, 1]
    ds = corr.src_scale / np.maximum(corr.dst_scale, 1e-8)
    dtheta = np.abs(np.arccos(np.clip(np.cos(corr.src_angle), -1, 1)) -
                     np.arccos(np.clip(np.cos(corr.dst_angle), -1, 1)))

    ok_dx = _hist_mask_threshold(dx, HIST_BINS, DXY_THRESH)
    ok_dy = _hist_mask_threshold(dy, HIST_BINS, DXY_THRESH)
    ok_ds = _hist_mask_mode(ds, HIST_BINS)
    ok_dtheta = _hist_mask_mode(dtheta, HIST_BINS)

    return (ok_dx & ok_dy) & (ok_dtheta | ok_ds)


# ---------------------------------------------------------------------------
# Global homography: base-paper DLT-eigen solve, plus RANSAC/MAGSAC wrappers
# ---------------------------------------------------------------------------

def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = homo @ H.T
    out = out[:, :2] / out[:, 2:3]
    return out


def _normalize_points(pts: np.ndarray):
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / max(mean_dist, 1e-8)
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    normed = (homo @ T.T)[:, :2]
    return normed, T


def fit_global_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """docx Sec 2.1: build DLT matrix A, solve A^T A h = lambda h (smallest
    eigenvalue eigenvector), reshape into 3x3 H. Points are Hartley-normalized
    first for numerical stability (algebraically equivalent, then un-normalized).
    """
    src_n, T_src = _normalize_points(src)
    dst_n, T_dst = _normalize_points(dst)

    n = len(src_n)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src_n[i]
        xp, yp = dst_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]

    # smallest-eigenvalue eigenvector of A^T A == smallest right singular vector of A
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_n = h.reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_n @ T_src
    return H / H[2, 2]


def fit_global_homography_cv(src: np.ndarray, dst: np.ndarray, method: int,
                              reproj_thresh: float = 3.0) -> np.ndarray:
    H, _ = cv2.findHomography(src, dst, method, reproj_thresh)
    if H is None:
        H = np.eye(3)
    return H


def warp_fn_from_homography(H: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda pts: apply_homography(H, pts)


# ---------------------------------------------------------------------------
# Mesh construction: Delaunay-over-matches and regular grid
# ---------------------------------------------------------------------------

def image_corners(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                     dtype=np.float64)


def build_delaunay_mesh(inlier_src_pts: np.ndarray, width: int, height: int):
    """docx Sec 1.4: vertices = N inlier positions + 4 corners, Delaunay over all."""
    corners = image_corners(width, height)
    vertices = np.vstack([inlier_src_pts, corners])
    tri = Delaunay(vertices)
    return vertices, tri.simplices, tri


def build_grid_mesh(width: int, height: int, nx: int = GRID_NX, ny: int = GRID_NY):
    """Regular grid mesh: (nx+1) x (ny+1) vertices, 2 triangles per cell with a
    fixed diagonal (TR-BL) so mesh-solve/render/point-locate all agree."""
    xs = np.linspace(0, width - 1, nx + 1)
    ys = np.linspace(0, height - 1, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.stack([xx.ravel(), yy.ravel()], axis=1)

    def vidx(i, j):
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            tl, tr, bl, br = vidx(i, j), vidx(i + 1, j), vidx(i, j + 1), vidx(i + 1, j + 1)
            tris.append([tl, tr, bl])
            tris.append([tr, br, bl])
    triangles = np.array(tris, dtype=np.int64)
    return vertices, triangles, (nx, ny, xs, ys)


class GridLocator:
    """Point-in-triangle locator for build_grid_mesh's fixed diagonal split."""

    def __init__(self, width, height, nx, ny):
        self.width, self.height, self.nx, self.ny = width, height, nx, ny
        self.cell_w = (width - 1) / nx
        self.cell_h = (height - 1) / ny

    def locate(self, pts: np.ndarray):
        x = np.clip(pts[:, 0], 0, self.width - 1 - 1e-6)
        y = np.clip(pts[:, 1], 0, self.height - 1 - 1e-6)
        i = np.clip((x / self.cell_w).astype(int), 0, self.nx - 1)
        j = np.clip((y / self.cell_h).astype(int), 0, self.ny - 1)
        fx = (x - i * self.cell_w) / self.cell_w
        fy = (y - j * self.cell_h) / self.cell_h

        def vidx(ii, jj):
            return jj * (self.nx + 1) + ii

        tl, tr, bl, br = vidx(i, j), vidx(i + 1, j), vidx(i, j + 1), vidx(i + 1, j + 1)
        upper = (fx + fy) <= 1.0  # triangle (tl, tr, bl)
        tri_idx = np.where(upper[:, None], np.stack([tl, tr, bl], axis=1),
                            np.stack([tr, br, bl], axis=1))
        return tri_idx


# ---------------------------------------------------------------------------
# Stage 1.5: ASAP constants + data+ASAP(+prior) sparse mesh solve
# ---------------------------------------------------------------------------

def compute_asap_constants(vertices: np.ndarray, triangles: np.ndarray):
    va = vertices[triangles[:, 0]]
    vb = vertices[triangles[:, 1]]
    vc = vertices[triangles[:, 2]]
    e = vb - va
    e_perp = np.stack([-e[:, 1], e[:, 0]], axis=1)
    elen2 = np.maximum((e ** 2).sum(axis=1), 1e-12)
    diff = vc - va
    alpha = (e * diff).sum(axis=1) / elen2
    beta = (e_perp * diff).sum(axis=1) / elen2
    return alpha, beta


def mesh_solve(vertices: np.ndarray, triangles: np.ndarray,
               data_idx: np.ndarray, data_targets: np.ndarray,
               w_data: float = W_DATA, w_asap: float = W_ASAP,
               prior_targets: Optional[np.ndarray] = None, w_prior: float = 0.0) -> np.ndarray:
    """docx Sec 1.5: solve for target vertex positions v' via one sparse LSQR
    over the stacked [x_1..x_Nv, y_1..y_Nv] unknown vector (x/y are coupled by
    the rot90 term in the ASAP constraint, so they cannot be solved separately).
    """
    nv = len(vertices)
    alpha, beta = compute_asap_constants(vertices, triangles)

    rows, cols, vals, rhs = [], [], [], []
    r = 0

    sw = np.sqrt(w_data)
    for k, (tx, ty) in zip(data_idx, data_targets):
        rows += [r, r + 1]; cols += [k, nv + k]; vals += [sw, sw]; rhs += [sw * tx, sw * ty]
        r += 2

    sa = np.sqrt(w_asap)
    a_idx, b_idx, c_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    for t in range(len(triangles)):
        a, b, c = a_idx[t], b_idx[t], c_idx[t]
        al, be = alpha[t], beta[t]
        # x-eq: x_c + (alpha-1) x_a - alpha x_b - beta y_a + beta y_b = 0
        rows += [r] * 5
        cols += [c, a, b, nv + a, nv + b]
        vals += [sa * 1, sa * (al - 1), sa * (-al), sa * (-be), sa * be]
        rhs.append(0.0)
        r += 1
        # y-eq: y_c + (alpha-1) y_a - alpha y_b + beta x_a - beta x_b = 0
        rows += [r] * 5
        cols += [nv + c, nv + a, nv + b, a, b]
        vals += [sa * 1, sa * (al - 1), sa * (-al), sa * be, sa * (-be)]
        rhs.append(0.0)
        r += 1

    if prior_targets is not None and w_prior > 0:
        sp = np.sqrt(w_prior)
        for k in range(nv):
            tx, ty = prior_targets[k]
            rows += [r, r + 1]; cols += [k, nv + k]; vals += [sp, sp]; rhs += [sp * tx, sp * ty]
            r += 2

    A = coo_matrix((vals, (rows, cols)), shape=(r, 2 * nv)).tocsr()
    b = np.array(rhs)
    sol = lsqr(A, b, atol=1e-10, btol=1e-10)[0]
    return np.stack([sol[:nv], sol[nv:]], axis=1)


def count_inverted_triangles(vertices: np.ndarray, solved: np.ndarray, triangles: np.ndarray) -> int:
    def signed_area(v, tris):
        a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
        e1, e2 = b - a, c - a
        return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    src_sign = np.sign(signed_area(vertices, triangles))
    dst_sign = np.sign(signed_area(solved, triangles))
    return int(np.sum(src_sign != dst_sign))


# ---------------------------------------------------------------------------
# Stage 1.6: barycentric + Hermite/Jacobian interpolation
# ---------------------------------------------------------------------------

def barycentric_weights(vertices: np.ndarray, tri_vertex_ids: np.ndarray,
                         pts: np.ndarray) -> np.ndarray:
    """Barycentric coords of `pts` w.r.t. their containing triangle, given as
    (N,3) vertex-id triples (the common locator convention used below)."""
    tris = tri_vertex_ids
    a, b, c = vertices[tris[:, 0]], vertices[tris[:, 1]], vertices[tris[:, 2]]
    v0, v1, v2 = b - a, c - a, pts - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    denom = np.maximum(d00 * d11 - d01 * d01, 1e-12)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.stack([u, v, w], axis=1)


def make_delaunay_locator(delaunay: Delaunay, triangles: np.ndarray):
    """Returns locate(pts) -> (N,3) vertex-id triples (the common convention)."""
    def locate(pts):
        tri_idx = delaunay.find_simplex(pts, tol=1e-6)
        # nudge any point that fell just outside the hull (float precision at edges)
        missing = tri_idx < 0
        if np.any(missing):
            hull_pts = delaunay.points
            centroid = hull_pts.mean(axis=0)
            nudged = pts[missing] + 1e-3 * (centroid - pts[missing])
            tri_idx[missing] = delaunay.find_simplex(nudged, tol=1e-6)
            tri_idx[tri_idx < 0] = 0
        return triangles[tri_idx]
    return locate


def barycentric_warp_fn(vertices, solved, locate_fn) -> Callable[[np.ndarray], np.ndarray]:
    def warp(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        tri_ids = locate_fn(pts)
        bary = barycentric_weights(vertices, tri_ids, pts)
        vabc = solved[tri_ids]  # (N,3,2)
        return (bary[:, :, None] * vabc).sum(axis=1)
    return warp


def compute_vertex_jacobians(n_vertices: int, data_idx: np.ndarray,
                              src_scale, dst_scale, src_angle, dst_angle) -> np.ndarray:
    """docx Sec 1.6: per-vertex local Jacobian from SIFT scale/orientation.
    J_k = (s_k'/s_k) * R(theta_k' - theta_k). Corner (non-data) vertices get identity.
    """
    J = np.tile(np.eye(2), (n_vertices, 1, 1))
    ratio = dst_scale / np.maximum(src_scale, 1e-8)
    dtheta = dst_angle - src_angle
    cos_t, sin_t = np.cos(dtheta), np.sin(dtheta)
    R = np.stack([np.stack([cos_t, -sin_t], axis=1),
                  np.stack([sin_t, cos_t], axis=1)], axis=1)
    J[data_idx] = ratio[:, None, None] * R
    return J


def hermite_warp_fn(vertices, solved, jacobians, locate_fn) -> Callable[[np.ndarray], np.ndarray]:
    """docx Sec 1.6: affine barycentric prediction + a per-vertex Hermite-blended
    Jacobian correction that vanishes exactly at each vertex."""
    def warp(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        tri_ids = locate_fn(pts)
        bary = barycentric_weights(vertices, tri_ids, pts)
        vabc = vertices[tri_ids]        # (N,3,2) source
        vabc2 = solved[tri_ids]         # (N,3,2) target
        Jabc = jacobians[tri_ids]       # (N,3,2,2)

        affine = (bary[:, :, None] * vabc2).sum(axis=1)  # (N,2)
        correction = np.zeros_like(affine)
        for k in range(3):
            offset = pts - vabc[:, k, :]                       # (N,2)
            E_k = vabc2[:, k, :] + np.einsum('nij,nj->ni', Jabc[:, k, :, :], offset)
            C_k = E_k - affine
            lam = bary[:, k]
            w_k = 3 * lam ** 2 - 2 * lam ** 3
            correction += w_k[:, None] * C_k
        return affine + correction
    return warp


# ---------------------------------------------------------------------------
# Stage 1.7: triangle-based rendering
# ---------------------------------------------------------------------------

def warp_triangle(src_img, dst_img, coverage, src_tri, dst_tri):
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    H, W = dst_img.shape[:2]
    x2c, y2c = max(x2, 0), max(y2, 0)
    w2c, h2c = min(x2 + w2, W) - x2c, min(y2 + h2, H) - y2c
    if w2c <= 0 or h2c <= 0:
        return

    src_off = (src_tri - [x1, y1]).astype(np.float32)
    dst_off = (dst_tri - [x2, y2]).astype(np.float32)
    src_crop = _safe_crop(src_img, x1, y1, w1, h1)
    M = cv2.getAffineTransform(src_off, dst_off)
    warped = cv2.warpAffine(src_crop, M, (w2, h2), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_off).astype(np.int32), 255)

    sub_mask = mask[y2c - y2:y2c - y2 + h2c, x2c - x2:x2c - x2 + w2c]
    sub_warp = warped[y2c - y2:y2c - y2 + h2c, x2c - x2:x2c - x2 + w2c]
    roi = dst_img[y2c:y2c + h2c, x2c:x2c + w2c]
    m = sub_mask > 0
    roi[m] = sub_warp[m]
    coverage[y2c:y2c + h2c, x2c:x2c + w2c][m] = True


def _safe_crop(img, x, y, w, h):
    H, W = img.shape[:2]
    canvas = np.zeros((h, w) + img.shape[2:], dtype=img.dtype)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        canvas[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return canvas


def render_via_triangles(img1: np.ndarray, out_shape, src_tris: np.ndarray, dst_tris: np.ndarray):
    """src_tris/dst_tris: (Nt,3,2) pixel-space triangle vertex arrays."""
    H, W = out_shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8) if img1.ndim == 3 else np.zeros((H, W), dtype=img1.dtype)
    coverage = np.zeros((H, W), dtype=bool)
    for i in range(len(src_tris)):
        warp_triangle(img1, canvas, coverage, src_tris[i], dst_tris[i])
    return canvas, coverage


def subdivide_barycentric(k: int):
    """Fixed barycentric grid of order k -> (points as (lam_a,lam_b,lam_c), sub-triangles)."""
    idx = {}
    pts = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            idx[(i, j)] = len(pts)
            pts.append((1 - (i + j) / k, i / k, j / k))
    tris = []
    for i in range(k):
        for j in range(k - i):
            tris.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if i + j < k - 1:
                tris.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    return np.array(pts), np.array(tris)


def render_affine_mesh(img1, out_shape, vertices, solved, triangles):
    """Piecewise-affine render: one affine warp per coarse triangle (exact for
    plain barycentric interpolation, used by CPW/li_tang/ours_barycentric)."""
    src_tris = vertices[triangles]
    dst_tris = solved[triangles]
    return render_via_triangles(img1, out_shape, src_tris, dst_tris)


def render_hermite_mesh(img1, out_shape, vertices, triangles, jacobians, solved, k: int = SUBDIVIDE_K):
    """Fine-subdivided render so the nonlinear Hermite correction is visible."""
    bary_pts, sub_tris = subdivide_barycentric(k)  # (P,3), (S,3)
    n_tri = len(triangles)
    P = len(bary_pts)

    va = vertices[triangles[:, 0]]; vb = vertices[triangles[:, 1]]; vc = vertices[triangles[:, 2]]
    va2 = solved[triangles[:, 0]]; vb2 = solved[triangles[:, 1]]; vc2 = solved[triangles[:, 2]]
    Ja = jacobians[triangles[:, 0]]; Jb = jacobians[triangles[:, 1]]; Jc = jacobians[triangles[:, 2]]

    lam = bary_pts  # (P,3)
    # source grid points per triangle: (n_tri, P, 2)
    src_grid = (lam[None, :, 0:1] * va[:, None, :] + lam[None, :, 1:2] * vb[:, None, :] +
                lam[None, :, 2:3] * vc[:, None, :])
    affine = (lam[None, :, 0:1] * va2[:, None, :] + lam[None, :, 1:2] * vb2[:, None, :] +
              lam[None, :, 2:3] * vc2[:, None, :])

    def vertex_term(v_k, v_k2, J_k, lam_k):
        offset = src_grid - v_k[:, None, :]                                  # (n_tri,P,2)
        E_k = v_k2[:, None, :] + np.einsum('tij,tpj->tpi', J_k, offset)      # (n_tri,P,2)
        C_k = E_k - affine
        w_k = 3 * lam_k[None, :] ** 2 - 2 * lam_k[None, :] ** 3
        return w_k[:, :, None] * C_k

    correction = (vertex_term(va, va2, Ja, lam[:, 0]) + vertex_term(vb, vb2, Jb, lam[:, 1]) +
                  vertex_term(vc, vc2, Jc, lam[:, 2]))
    dst_grid = affine + correction  # (n_tri, P, 2)

    src_flat = src_grid.reshape(n_tri * P, 2)
    dst_flat = dst_grid.reshape(n_tri * P, 2)
    offsets = (np.arange(n_tri) * P)[:, None]
    sub_tris_all = (sub_tris[None, :, :] + offsets[:, None, :]).reshape(-1, 3)

    src_tris = src_flat[sub_tris_all]
    dst_tris = dst_flat[sub_tris_all]
    return render_via_triangles(img1, out_shape, src_tris, dst_tris)


def timer():
    return time.perf_counter()
