# SPATIAL-X v3 user guide

## The four things you see

- **Object masks:** colored overlays show the pixels assigned to each detected object. A label such as `Object 2: chair` identifies the instance in this image. Detection confidence is a model score, not guaranteed correctness.
- **Relative depth:** the display visualizes the depth model's prediction for this image. It is normalized for display and is not a distance measurement in metres.
- **3D point cloud:** pixels are projected into 3D-like coordinates using estimated depth and approximate camera intrinsics. Rotate/zoom the chart. Choose instance colors to inspect segmentation transfer, or original RGB to see source colors.
- **Text retrieval:** CLIP compares each detected crop with a text prompt and ranks them by cosine similarity. It can only rank objects already detected; it is not a text-prompted segmenter. Scores are not probabilities.

## Read the graphs

- **Confidence chart:** longer bars mean higher detector scores, not necessarily more correct objects.
- **Text ranking chart:** higher cosine similarity means a closer embedding match among the shown crops. Compare candidates within the same query; do not treat it as a calibrated probability.
- **Latency chart:** time in seconds for each measured stage. First-run timing may include model initialization/download overhead. For a fair comparison, record hardware, resolution, warm/cold status and repeated runs.

## A sensible first experiment

1. Use a well-lit indoor image with several visible objects.
2. Keep the default threshold and run with depth enabled, CLIP disabled.
3. Inspect the masks and the depth visualization; note missed objects and ambiguous boundaries.
4. Inspect the point cloud and remember that the coordinate scale is approximate.
5. Turn on CLIP and ask a simple query like `chair`; inspect the ranking.
6. Save the reproducibility bundle and the run-summary JSON.
7. Repeat with a second image, then compare results only after documenting the conditions.

## What is not yet measured

Without aligned ground truth and a dataset protocol, this demo does not measure metric-depth accuracy, benchmark instance AP, KITTI 3D AP, occupancy mIoU, or scientific reconstruction quality. The UI labels these as unmeasured rather than showing misleading zero scores.
