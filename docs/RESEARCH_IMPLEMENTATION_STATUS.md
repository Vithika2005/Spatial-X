# SPATIAL-X v3: implementation status and paper-correctness plan

## Why this document exists

SPATIAL-X contains a runnable prototype plus a planned research suite. These are not equivalent. A paper method is considered reproduced only after the method's official architecture/training or inference procedure, required data preprocessing, checkpoint provenance, and official evaluation protocol have been verified. A similar-looking output is not evidence of reproduction.

## Current runnable path

1. **YOLOv8-seg pretrained baseline**: returns 2D boxes, class names, confidence scores, and instance masks.
2. **MiDaS depth estimation (optional)**: produces an image-relative depth estimate. Its output must not be interpreted as metres without a valid calibration/scale procedure.
3. **Geometric lifting**: back-projects pixels using guessed focal length and image center. The resulting coordinates are pseudo-3D and are for visualization only.
4. **CLIP ViT-B/32 crop/text retrieval (optional)**: compares normalized image and text embeddings with cosine similarity, then ranks only the object crops that the detector found. Scores are not probabilities.
5. **Exports and instrumentation**: saves available artifacts and run metadata. Ground-truth metrics remain `not_evaluated` until a proper dataset evaluator is run.

## Method-by-method status

| Paper / track | Current status | What is needed for a correct implementation or reproduction |
|---|---|---|
| Segment Anything / SAM 2 | Not integrated | Use official model/checkpoints, prompt design, video memory/propagation for SAM 2, then compare mask quality with annotated ground truth. Current masks are YOLOv8-seg. |
| Splatter Image (CVPR 2024) | Not implemented | Follow official architecture/checkpoints and supported object dataset; render predicted Gaussian representation; evaluate with the paper's reconstruction/rendering protocol. The current point cloud is not a Gaussian scene. |
| LERF (ICCV 2023) | Not implemented | Train/use the language-embedded radiance field with multi-view images and poses following official setup; evaluate spatial language queries with a supported scene benchmark. CLIP crop ranking is only a baseline. |
| MonoDETR (ICCV 2023) / MonoDTR (CVPR 2022) | Not implemented | Choose one, reproduce official preprocessing/config/checkpoint; use KITTI calibration/labels and official 3D AP protocol. Current detector produces 2D boxes/masks, not oriented 3D boxes. |
| VoxFormer (CVPR 2023) | Not implemented | Implement official sparse voxel transformer and scene completion data pipeline; evaluate occupancy/semantic completion on the supported benchmark. A point cloud is not occupancy completion. |
| 4D Gaussian Splatting (CVPR 2024) | Not implemented | Follow official dynamic scene pipeline on calibrated multi-frame sequences; evaluate rendering quality and temporal behavior. Selecting a frame from a video is not 4D reconstruction. |
| ScanNet / ScanNet++ | Dataset options, not methods | Obtain access under dataset terms, align RGB-D, intrinsics, poses and labels, and follow an explicit split/protocol. |
| KITTI | Dataset/benchmark option, not a method | Use official split, calibration, labels and evaluation for monocular 3D detection. |

## Recommended sequence

### Stage A — Reliable baseline (Mac-first)
- Fix package versions and record the exact checkpoint names/revisions.
- Add small smoke tests and document cold-start versus warm-start timing.
- Keep relative-depth and pseudo-3D caveats visible in the UI and exported metadata.
- Validate that CLIP similarities are cosine similarities, not a softmax over the current image's crops.

### Stage B — Dataset-backed evaluation
- Start with a small, permitted ScanNet subset.
- Build a manifest with scene/frame IDs, RGB path, depth path, intrinsics, pose, and available annotations.
- Align predicted and ground-truth depth, define invalid-pixel and scale policies, and report Abs Rel, RMSE, and delta thresholds.
- Compare masks only where compatible instance labels exist. Report mask IoU/AP using an explicitly documented matching protocol.
- For text retrieval, create a held-out list of queries and correct instance IDs; report Recall@1/5 and confidence intervals where sample size supports them.

### Stage C — Reproduce one paper at a time

Do not attempt all heavy tracks simultaneously. Select one based on the intended research question and compute access. Use the official repository as the source of truth; record commit SHA, dependencies, checkpoints, dataset split, training/inference command, and any deviations. Reproduce the paper's own metric first before proposing a cross-method comparison.

Suggested order for this Mac-first project:
1. SAM 2 as a mask-generation/video extension if supported by available hardware and packages.
2. A ScanNet RGB-D geometry/instance-lifting evaluation as the primary controlled experiment.
3. One of LERF or Splatter Image on its supported multi-view/object benchmark, potentially using a compatible remote Linux GPU if official CUDA kernels require it.
4. KITTI MonoDETR/MonoDTR and VoxFormer as separate benchmark tracks, not add-ons to the indoor ScanNet metric table.
5. 4D Gaussian Splatting only when a calibrated sequence and compatible renderer are available.

## Claims checklist for paper, CV, or presentation

- Say **uses a pretrained baseline** when running an off-the-shelf checkpoint.
- Say **adapted** only when you changed a method and document the changes.
- Say **reproduced** only after following and verifying the original method and protocol.
- Say **proposed** for a new contribution, and support it with controlled experiments/ablations.
- Never compare KITTI AP3D, ScanNet mask IoU, depth Abs Rel, and CLIP cosine similarity as if they were the same metric.
- Do not fill missing benchmark metrics with zero; report `not evaluated`.
