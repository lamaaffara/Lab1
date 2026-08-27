# Results summary

Averaged over 10 synthetic seeds (0-9), see `comparison_table.csv` for the full method x metric table (mean +/- std) and `comparison_grid.png` for a representative-seed render of every method's warp plus its pixel-error heatmap.

On dense random query points, **MAGSAC++ + global H** has the lowest mean geometric error (3.47px) and **Ours: Delaunay, Hermite (best)** the highest (7.14px); **MAGSAC++ + global H** is the fastest outlier-removal-plus-warp-estimation stage (0.59ms) and **APAP / Moving-DLT** the slowest (164.90ms), which matches APAP's O(grid_cells x matches) weighted-DLT-per-cell cost against RANSAC/MAGSAC's native OpenCV implementations and the O(N) single sparse solve of the mesh methods.

Flag: all Delaunay/grid mesh methods extrapolate poorly at the 4 image corners by construction (no data term reaches them, docx Sec 6.2) -- their mean corner error is 2.3x the global-homography methods' corner error on average, which is expected, not a bug. More notable: on this synthetic bump+vortex protocol our mesh methods (mean dense-point error 6.71px) do NOT clearly beat the global-homography baselines (mean 4.02px) averaged across seeds -- counter to the single-image internal results the docx flags as unvalidated (Sec 7.3). The relative-attribute filter also consistently keeps far fewer correspondences (58-439 inliers across seeds/methods) than RANSAC/MAGSAC's own inlier sets, which plausibly explains part of the gap: fewer, noisy pinned mesh vertices give the ASAP solve less to work with than a global model fit from many more correspondences.

11 inverted (fold-over) triangles were observed in total across all mesh-method/seed runs -- worth a closer look before trusting mesh output on those specific seeds.

**TODO:** no real image-pair dataset (GES/UDIS-D) is wired in -- there's no public sample of either reachable without auth/scraping setup within this project's time budget, so `data.load_real_pair()` exists as a loader stub but is unused. Synthetic coverage across seeds is the primary evidence here, per the task's own guidance that seed coverage matters more than real-dataset coverage for this pass.
