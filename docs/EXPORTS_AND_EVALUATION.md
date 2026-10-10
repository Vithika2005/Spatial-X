# Reproducible Exports and Evaluation

Click **Save reproducibility bundle to outputs/** in the Streamlit UI. For `scene.jpg`, artifacts include:

- `scene_overlay.png`: masks, labels, boxes.
- `scene_depth.png`: normalized depth visualization (not metric depth).
- `scene_depth_raw.npy`: raw model depth array.
- `scene_pointcloud.ply`: portable ASCII PLY with XYZ, RGB, and instance ID.
- `scene_pointcloud.npz`: compact NumPy arrays `xyz`, `rgb`, `instance_id`.
- `scene_pointcloud.csv`: inspectable point table.
- `scene_query_scores.csv`: prompt, instance ID, class, detector confidence, and CLIP similarity when a query ran.
- `scene_run_metadata.json`: software versions, input shape, configuration, timings, detections, query scores, point count, and metric status.
- `scene_evaluation_metrics.json`: evaluation metrics, or explicit `null` values and `not_evaluated` status until ground truth is used.

## Timing interpretation
`segmentation_seconds` includes model loading if the run is cold. `postprocess_and_overlay_seconds` measures mask/box conversion and overlay drawing. `depth_seconds` includes depth model loading if cold. `point_cloud_seconds` and `ui_total_wall_seconds` are recorded by the app. Compare warm runs separately from first-run startup and report median/p95 across repeated runs. CLIP query latency is not yet separately profiled in this version; do not present the placeholder as a measured value.

## Metrics and dataset validity
A prediction alone cannot yield accuracy. `spatialx/evaluation.py` includes helpers for aligned depth metrics, binary mask IoU, and retrieval Recall@K. It is not a replacement for the official ScanNet or KITTI evaluator. Use official splits, label mappings, camera intrinsics, poses, and evaluator. Keep uncomputed metrics null rather than writing zero.

## Geometry caveat
The current monocular depth path plus approximate focal length yields pseudo-3D. Its coordinates are not meters and are not aligned to ScanNet world coordinates. For metric geometry, use calibrated RGB-D, matching intrinsics and poses, then transform camera-frame points into the official world coordinate frame.
