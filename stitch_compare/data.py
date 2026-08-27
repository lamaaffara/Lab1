"""Synthetic ground-truth generator (docx Sec 6.1) + a trivial real-pair loader.

Synthetic protocol: start from a real photo, apply a KNOWN random homography,
then compose it with a smooth non-homographic bump+vortex perturbation field,
so the true pixel-level warp from image 1 to image 2 is known exactly
everywhere but is not expressible by any single homography.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from common import apply_homography, image_corners

# Bundled, offline scikit-image sample photos used as the "real photo" starting
# point, cycled by seed so results aren't reported from a single source image
# (the docx explicitly flags n=1 prior results as insufficient).
SAMPLE_IMAGES = ["astronaut", "coffee", "chelsea", "camera", "rocket", "hubble_deep_field",
                  "cat", "brick"]

IMG_SIZE = 480


def load_sample_image(seed: int, size: int = IMG_SIZE) -> np.ndarray:
    import skimage.data as skd
    names = [n for n in SAMPLE_IMAGES if hasattr(skd, n)]
    name = names[seed % len(names)]
    img = getattr(skd, name)()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def random_homography(rng: np.random.Generator, size: int, jitter_frac: float = 0.08) -> np.ndarray:
    src = image_corners(size, size).astype(np.float32)
    jitter = rng.uniform(-1, 1, size=(4, 2)) * jitter_frac * size
    dst = (src + jitter).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H.astype(np.float64)


def bump_vortex_field(pts: np.ndarray, center_bump, amp_bump, sigma_bump,
                       center_vortex, amp_vortex, sigma_vortex) -> np.ndarray:
    """Smooth, small-amplitude non-homographic displacement field: a radial
    Gaussian 'bump' plus a rotational 'vortex' term (docx Sec 6.1)."""
    d_b = pts - np.asarray(center_bump)
    r2_b = (d_b ** 2).sum(axis=1)
    g_b = amp_bump * np.exp(-r2_b / (2 * sigma_bump ** 2))
    dir_b = d_b / np.maximum(np.linalg.norm(d_b, axis=1, keepdims=True), 1e-6)
    disp_bump = dir_b * g_b[:, None]

    d_v = pts - np.asarray(center_vortex)
    r2_v = (d_v ** 2).sum(axis=1)
    g_v = amp_vortex * np.exp(-r2_v / (2 * sigma_vortex ** 2))
    tang = np.stack([-d_v[:, 1], d_v[:, 0]], axis=1)
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    disp_vortex = tang * g_v[:, None]

    return disp_bump + disp_vortex


@dataclass
class SyntheticPair:
    seed: int
    img1: np.ndarray
    img2: np.ndarray
    H: np.ndarray                 # ground-truth homography component
    field_params: dict            # bump/vortex parameters
    ground_truth_map: Callable[[np.ndarray], np.ndarray]  # exact image1 -> image2 map
    size: int


def make_synthetic_pair(seed: int, size: int = IMG_SIZE) -> SyntheticPair:
    rng = np.random.default_rng(seed)
    img1 = load_sample_image(seed, size)
    H = random_homography(rng, size)

    margin = 0.25 * size
    field_params = dict(
        center_bump=rng.uniform(margin, size - margin, size=2),
        amp_bump=rng.uniform(6, 14),
        sigma_bump=rng.uniform(0.12, 0.25) * size,
        center_vortex=rng.uniform(margin, size - margin, size=2),
        amp_vortex=rng.uniform(4, 10),
        sigma_vortex=rng.uniform(0.12, 0.25) * size,
    )

    def ground_truth_map(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        hpts = apply_homography(H, pts)
        disp = bump_vortex_field(hpts, **field_params)
        return hpts + disp

    # Render image2 by inverse-warping image1. The perturbation is small-amplitude
    # by construction, so a zeroth-order inverse (invert H, then subtract the
    # field evaluated near the destination point) is an accurate enough
    # approximation for rendering texture; the *exact* forward map above (not
    # this rendering shortcut) is what evaluate.py scores methods against.
    ys, xs = np.mgrid[0:size, 0:size]
    dst_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    Hinv = np.linalg.inv(H)
    disp_at_dst = bump_vortex_field(dst_pts, **field_params)
    src_est = apply_homography(Hinv, dst_pts - disp_at_dst)
    map_x = src_est[:, 0].reshape(size, size).astype(np.float32)
    map_y = src_est[:, 1].reshape(size, size).astype(np.float32)
    img2 = cv2.remap(img1, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)

    return SyntheticPair(seed, img1, img2, H, field_params, ground_truth_map, size)


def sample_near_perturbation_points(pair: SyntheticPair, n: int = 1000,
                                     seed_offset: int = 1000) -> np.ndarray:
    """Sample source-image (image 1) points whose warped position lands near
    the bump/vortex centers. The field is defined in post-homography
    (image 2) space (see ground_truth_map: it's evaluated at H(pts)), so the
    region of image 1 that actually stress-tests the perturbation is the
    PRE-IMAGE of each center under H, not the center coordinates themselves.
    """
    rng = np.random.default_rng(pair.seed + seed_offset)
    n_each = n // 2
    fp = pair.field_params
    Hinv = np.linalg.inv(pair.H)
    src_center_bump = apply_homography(Hinv, fp["center_bump"][None, :])[0]
    src_center_vortex = apply_homography(Hinv, fp["center_vortex"][None, :])[0]
    pts_b = rng.normal(src_center_bump, fp["sigma_bump"] * 0.8, size=(n_each, 2))
    pts_v = rng.normal(src_center_vortex, fp["sigma_vortex"] * 0.8, size=(n - n_each, 2))
    pts = np.vstack([pts_b, pts_v])
    return np.clip(pts, 0, pair.size - 1)


def sample_dense_random_points(pair: SyntheticPair, n: int = 3000,
                                seed_offset: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(pair.seed + seed_offset)
    return rng.uniform(0, pair.size - 1, size=(n, 2))


# ---------------------------------------------------------------------------
# Real image pair loader (trivial stub only -- see results/README.md TODO).
# ---------------------------------------------------------------------------

def load_real_pair(path1: str, path2: str):
    """Load a real (img1, img2) pair from disk. No ground-truth warp is known
    for real pairs, so this is usable only for qualitative rendering / the
    MGE intensity-agreement metric, never the geometric-error metrics.

    No public GES/UDIS-D sample pairs are fetched here -- see the TODO in
    results/README.md. This loader is provided so real pairs can be dropped
    in later with zero code changes.
    """
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read real image pair: {path1}, {path2}")
    return img1, img2
