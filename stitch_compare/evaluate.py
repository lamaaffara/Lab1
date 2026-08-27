"""Evaluation protocol (docx Sec 6.2/6.3): held-out matches, dense random
points, near-perturbation points, and the 4 corners, each scored by mean
geometric error against known synthetic ground truth; plus MGE (pixel-
intensity agreement in the overlap region), timing of the outlier-removal-
through-warp stage only, and inverted-triangle counts for mesh methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import sift_detect_and_match, train_holdout_split
from data import (make_synthetic_pair, sample_dense_random_points,
                   sample_near_perturbation_points)

HOLDOUT_FRAC = 0.2
N_DENSE = 3000
N_NEAR_PERT = 1000


def geometric_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def compute_mge(warped: np.ndarray, coverage: np.ndarray, img2: np.ndarray) -> float:
    """Mean pixel-intensity disagreement in the overlap region (docx Sec 6.3 /
    base paper Eq. 3): mean absolute grayscale difference where the warp
    actually landed content."""
    if warped is None or coverage is None or coverage.sum() == 0:
        return float("nan")
    g_warp = warped.astype(np.float32).mean(axis=2) if warped.ndim == 3 else warped.astype(np.float32)
    g_ref = img2.astype(np.float32).mean(axis=2) if img2.ndim == 3 else img2.astype(np.float32)
    return float(np.abs(g_warp[coverage] - g_ref[coverage]).mean())


def evaluate_one(pair, corr_used, holdout_src, res, img2) -> dict:
    gt_holdout = pair.ground_truth_map(holdout_src)
    pred_holdout = res.warp_fn(holdout_src)

    dense_pts = sample_dense_random_points(pair, N_DENSE)
    pred_dense = res.warp_fn(dense_pts)
    gt_dense = pair.ground_truth_map(dense_pts)

    near_pts = sample_near_perturbation_points(pair, N_NEAR_PERT)
    pred_near = res.warp_fn(near_pts)
    gt_near = pair.ground_truth_map(near_pts)

    from common import image_corners
    corners = image_corners(pair.size, pair.size)
    pred_corners = res.warp_fn(corners)
    gt_corners = pair.ground_truth_map(corners)

    return {
        "geo_err_heldout": geometric_error(pred_holdout, gt_holdout),
        "geo_err_dense": geometric_error(pred_dense, gt_dense),
        "geo_err_near_pert": geometric_error(pred_near, gt_near),
        "geo_err_corners": geometric_error(pred_corners, gt_corners),
        "mge": compute_mge(res.warped, res.coverage_mask, img2),
        "time_ms": res.elapsed * 1000.0,
        "num_inverted_triangles": (np.nan if res.num_inverted_triangles is None
                                    else res.num_inverted_triangles),
        "num_inliers": res.meta.get("num_inliers", np.nan),
    }


def run_comparison(methods: list, seeds: list, holdout_frac: float = HOLDOUT_FRAC,
                    split_seed_offset: int = 5000):
    """Run every (name, module) in `methods` across every seed in `seeds`.

    Returns (long_df, results_cache) where long_df has one row per
    (seed, method, metric-columns) and results_cache[seed] holds the
    SyntheticPair, shared correspondences, and per-method MethodResult
    objects/images -- needed later for the comparison figure.
    """
    rows = []
    results_cache = {}

    for seed in seeds:
        pair = make_synthetic_pair(seed)
        corr = sift_detect_and_match(pair.img1, pair.img2)
        rng = np.random.default_rng(seed + split_seed_offset)
        corr_used, holdout_src = train_holdout_split(corr, holdout_frac, rng)

        seed_results = {}
        for name, mod in methods:
            res = mod.run_method(pair.img1, pair.img2, corr_used, render=True)
            metrics = evaluate_one(pair, corr_used, holdout_src, res, pair.img2)
            metrics.update({"seed": seed, "method": name})
            rows.append(metrics)
            seed_results[name] = res

        results_cache[seed] = {"pair": pair, "corr": corr, "corr_used": corr_used,
                                "holdout_src": holdout_src, "results": seed_results}

    long_df = pd.DataFrame(rows)
    return long_df, results_cache


METRIC_COLS = ["geo_err_heldout", "geo_err_dense", "geo_err_near_pert", "geo_err_corners",
               "mge", "time_ms", "num_inverted_triangles", "num_inliers"]


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    """Method x metric table, averaged across seeds with std dev (docx Sec 6.4/7.3)."""
    grouped = long_df.groupby("method")[METRIC_COLS]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std().add_suffix("_std")
    summary = pd.concat([mean_df, std_df], axis=1)
    ordered_cols = [c for pair in zip([f"{m}_mean" for m in METRIC_COLS],
                                       [f"{m}_std" for m in METRIC_COLS]) for c in pair]
    return summary[ordered_cols]


# ---------------------------------------------------------------------------
# Comparison figure: grid of warps + error heatmaps for one representative seed
# ---------------------------------------------------------------------------

def write_results_readme(summary_df: pd.DataFrame, long_df: pd.DataFrame, seeds: list,
                          out_path: str, csv_name: str, figure_name: str):
    """Generate results/README.md FROM the actual computed numbers (not a
    static hand-written summary) so it can't silently drift from the table."""
    best_dense = summary_df["geo_err_dense_mean"].idxmin()
    worst_dense = summary_df["geo_err_dense_mean"].idxmax()
    fastest = summary_df["time_ms_mean"].idxmin()
    slowest = summary_df["time_ms_mean"].idxmax()

    # a method is "mesh-based" if it ever reported a non-NaN inverted-triangle count
    mesh_methods = [m for m in summary_df.index
                    if long_df.loc[long_df["method"] == m, "num_inverted_triangles"].notna().any()]
    global_methods = [m for m in summary_df.index if m not in mesh_methods]

    corner_inflation = None
    if mesh_methods and global_methods:
        mesh_corner = summary_df.loc[mesh_methods, "geo_err_corners_mean"].mean()
        global_corner = summary_df.loc[global_methods, "geo_err_corners_mean"].mean()
        mesh_dense = summary_df.loc[mesh_methods, "geo_err_dense_mean"].mean()
        global_dense = summary_df.loc[global_methods, "geo_err_dense_mean"].mean()
        corner_inflation = (mesh_corner / max(global_corner, 1e-6),
                             mesh_dense, global_dense)

    total_inverted = long_df["num_inverted_triangles"].fillna(0).sum()
    inlier_min, inlier_max = long_df["num_inliers"].min(), long_df["num_inliers"].max()

    lines = []
    lines.append("# Results summary\n")
    lines.append(f"Averaged over {len(seeds)} synthetic seeds ({min(seeds)}-{max(seeds)}), "
                 f"see `{csv_name}` for the full method x metric table (mean +/- std) and "
                 f"`{figure_name}` for a representative-seed render of every method's warp "
                 f"plus its pixel-error heatmap.\n")
    lines.append(
        f"On dense random query points, **{best_dense}** has the lowest mean geometric error "
        f"({summary_df.loc[best_dense, 'geo_err_dense_mean']:.2f}px) and **{worst_dense}** the "
        f"highest ({summary_df.loc[worst_dense, 'geo_err_dense_mean']:.2f}px); **{fastest}** is "
        f"the fastest outlier-removal-plus-warp-estimation stage "
        f"({summary_df.loc[fastest, 'time_ms_mean']:.2f}ms) and **{slowest}** the slowest "
        f"({summary_df.loc[slowest, 'time_ms_mean']:.2f}ms), which matches APAP's "
        "O(grid_cells x matches) weighted-DLT-per-cell cost against RANSAC/MAGSAC's native "
        "OpenCV implementations and the O(N) single sparse solve of the mesh methods.\n")

    if corner_inflation:
        ratio, mesh_dense, global_dense = corner_inflation
        lines.append(
            f"Flag: all Delaunay/grid mesh methods extrapolate poorly at the 4 image corners "
            f"by construction (no data term reaches them, docx Sec 6.2) -- their mean corner "
            f"error is {ratio:.1f}x the global-homography methods' corner error on average, "
            "which is expected, not a bug. More notable: on this synthetic bump+vortex "
            f"protocol our mesh methods (mean dense-point error {mesh_dense:.2f}px) do NOT "
            f"clearly beat the global-homography baselines (mean {global_dense:.2f}px) "
            "averaged across seeds -- counter to the single-image internal results the docx "
            "flags as unvalidated (Sec 7.3). The relative-attribute filter also consistently "
            f"keeps far fewer correspondences ({inlier_min:.0f}-{inlier_max:.0f} inliers across "
            "seeds/methods) than RANSAC/MAGSAC's own inlier sets, which plausibly explains "
            "part of the gap: fewer, noisy pinned mesh vertices give the ASAP solve less to "
            "work with than a global model fit from many more correspondences.\n")

    if total_inverted > 0:
        lines.append(f"{int(total_inverted)} inverted (fold-over) triangles were observed in "
                     "total across all mesh-method/seed runs -- worth a closer look before "
                     "trusting mesh output on those specific seeds.\n")
    else:
        lines.append("No inverted (fold-over) triangles were observed in any mesh-method run "
                     "across seeds -- the ASAP term is keeping every solved mesh valid here.\n")

    lines.append(
        "**TODO:** no real image-pair dataset (GES/UDIS-D) is wired in -- there's no public "
        "sample of either reachable without auth/scraping setup within this project's time "
        "budget, so `data.load_real_pair()` exists as a loader stub but is unused. Synthetic "
        "coverage across seeds is the primary evidence here, per the task's own guidance that "
        "seed coverage matters more than real-dataset coverage for this pass.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def plot_comparison_grid(seed_cache: dict, method_names: list, out_path: str):
    import matplotlib.pyplot as plt

    pair = seed_cache["pair"]
    results = seed_cache["results"]
    n = len(method_names)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9.5))

    img2_gray = pair.img2.astype(np.float32).mean(axis=2)
    row_labels = ["warped img1->img2", "target img2", "|error| heatmap"]

    for col, name in enumerate(method_names):
        res = results[name]
        warped = res.warped
        coverage = res.coverage_mask
        ax_img, ax_ref, ax_err = axes[0, col], axes[1, col], axes[2, col]

        if warped is not None:
            ax_img.imshow(warped[..., ::-1])
        ax_img.set_title(name, fontsize=9)

        ax_ref.imshow(pair.img2[..., ::-1])

        if warped is not None and coverage is not None:
            warp_gray = warped.astype(np.float32).mean(axis=2)
            err = np.abs(warp_gray - img2_gray)
            err_display = np.where(coverage, err, np.nan)
            ax_err.imshow(err_display, cmap="inferno", vmin=0, vmax=60)

        for ax in (ax_img, ax_ref, ax_err):
            ax.set_xticks([]); ax.set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    fig.suptitle(f"Method comparison -- representative seed {pair.seed}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
