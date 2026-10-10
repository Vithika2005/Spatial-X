# SPATIAL-X — Detailed Project Plan

## 1. Problem statement

Common 2D detectors identify objects in the image plane but do not directly provide a coherent spatial representation. 3D perception methods can estimate geometry, while vision-language models can retrieve objects using text, but joining these outputs creates challenges around calibration, occlusion, scale, and cross-view consistency.

SPATIAL-X integrates instance segmentation, monocular depth estimation, geometric back-projection, and text-guided instance ranking into a reproducible demo. The research task is to measure how these components affect object-level spatial understanding.

## 2. Objectives

1. Detect object instances and obtain per-instance masks.
2. Estimate a depth map for each frame.
3. Lift masked image pixels into a colored 3D point cloud.
4. associate 3D points with instance IDs.
5. Rank object crops against natural-language queries.
6. provide a browser-based interface and exportable artifacts.
7. Evaluate segmentation, depth, retrieval, runtime, and failure cases.
8. Reproduce one advanced paper module as time permits, documenting differences from the official method.

## 3. Scope boundaries

The baseline does not itself reproduce LERF, Splatter Image, MonoDTR, VoxFormer, or 4D-GS. It uses existing pretrained models and a geometric projection stage. These methods solve related but different problems:
- LERF: language-grounded radiance fields.
- Splatter Image: feed-forward single-view 3D object reconstruction using Gaussians.
- MonoDTR/MonoDETR: monocular 3D bounding-box detection in driving scenes.
- VoxFormer: semantic scene completion/occupancy.
- 4D-GS: dynamic scene rendering over time.
- SAM/SAM 2: promptable image/video masks, not semantic detection by themselves.

## 4. Proposed research novelty (validate, do not assume)

**Instance-aware cross-modal lifting with confidence-aware filtering.** Each image pixel is assigned depth, RGB, and an instance ID. The 3D cloud supports text-ranked instance selection. Add an uncertainty/confidence term using detector confidence and depth consistency; compare with a baseline that uses no filtering.

Candidate hypotheses:
- H1: Removing low-confidence depth pixels improves geometric consistency at the cost of point coverage.
- H2: CLIP ranking improves open-vocabulary retrieval for categories not represented by the detector's fixed labels, but may fail on fine-grained attributes.
- H3: Lower resolution reduces latency but harms small-object segmentation and thin structures.
- H4: Occlusion and reflective/transparent surfaces disproportionately reduce depth and mask quality.

## 5. Architecture

### A. Input
Image upload or video frame selection. For reproducibility, record image dimensions, frame number, and resize configuration.

### B. Instance segmentation
YOLO segmentation baseline produces:
- bounding box
- class ID/name
- confidence
- binary instance mask

Upgrade option: use SAM 2 prompted by detector boxes to refine masks and propagate instances through a video. Track identity association separately; segmentation alone does not guarantee stable IDs across frames.

### C. Depth
The optional DPT/MiDaS pipeline estimates relative depth. It is not metric depth and may use an inverse depth convention. Never label it in meters. For ScanNet RGB-D evaluation, use ground-truth depth with calibration and pose.

### D. 2D-to-3D lifting
For each pixel (u,v) with depth z and intrinsics K:
X = (u - cx) z / fx
Y = (v - cy) z / fy
Z = z
A camera pose T_wc transforms camera-frame points into world coordinates:
p_world = T_wc * p_camera
For monocular relative depth, the result is a pseudo point cloud; unknown scale and approximate intrinsics prevent metric claims.

### E. Language retrieval
CLIP embeds the text prompt and detected object crops. Cosine/logit similarity ranks the instances. This is not equivalent to LERF: LERF trains a language field inside a radiance field and produces 3D query relevancy.

### F. UI and outputs
Streamlit displays masks, depth, point cloud, instance table, query ranking, and export paths.

## 6. Evaluation design

Create a fixed evaluation set with:
- diverse object classes
- small and large objects
- occlusion levels
- lighting conditions
- reflective/transparent objects
- motion blur for video

Maintain a CSV manifest with sample ID, source, split, scene ID, and ground-truth availability. Avoid train/test leakage across frames from the same scene.

### Metrics
- Segmentation: mask AP / IoU.
- Detection: mAP, precision, recall.
- Depth: Abs Rel, RMSE, δ thresholds against aligned ground truth.
- 3D geometry: Chamfer distance or point-to-surface error only when a valid ground-truth geometry and alignment exist.
- 3D semantics: per-class IoU/mIoU.
- Retrieval: Recall@1, Recall@5, MRR, with query labels.
- Efficiency: median and p95 latency, FPS, memory, input size, device, package versions.

### Baselines / ablations
1. Detector class only vs CLIP-ranked crop retrieval.
2. Ground-truth depth vs monocular depth (on aligned RGB-D scenes).
3. All points vs confidence-filtered points.
4. 320/416/512/640 input resolution.
5. Point caps: 5k/10k/20k/40k.
6. Clean image vs blur/low-light/reflection subsets.
7. RGB-only 2D baseline vs instance-aware 3D lifting.

Run at least 3 repetitions for timing; report median and p95. Do not compare latency across machines without listing hardware and software versions.

## 7. Dataset protocol

### ScanNet
Use official access procedure and terms. Start with a small approved subset. It contains RGB-D sequences, camera poses, reconstructions, and instance annotations. Preserve official split definitions. RGB/depth frames must be aligned and the intrinsic/extrinsic matrices interpreted correctly.

### KITTI
Use for the 8E extension. It contains camera images and LiDAR point clouds/3D labels for official 3D detection evaluation. MonoDTR and MonoDETR are not interchangeable with indoor ScanNet evaluation; they are driving-scene models.

### ScanNet++
High quality but extremely large. Use only if storage, download authorization, and compute are sufficient.

### Replica
Suitable for controlled indoor scene reconstruction. Confirm the license/terms and the representation needed by the chosen method.

## 8. Mac M-series execution strategy

- Start with Python 3.11 and the baseline.
- Use PyTorch MPS only when the exact model and operators support it.
- The app's segmentation model may choose its own device; check logs and actual runtime rather than assuming all inference is on Metal.
- The Transformers depth pipeline in this scaffold uses a conservative CPU fallback unless CUDA is available. This is intentional reliability, not an MPS performance claim.
- CLIP in this scaffold is CPU by default; modify to a tested MPS path and compare correctness before benchmarking.
- Official 3D Gaussian Splatting, 4D-GS, VoxFormer, and 3D detection repositories may rely on CUDA extensions. Use native Mac for the baseline and consider a compatible Linux/NVIDIA runtime for those specific paper reproductions.
- Record thermal/power context if reporting energy; don't equate GPU utilization with energy consumption.

## 9. Milestones

M1 environment, M2 2D segmentation, M3 depth and 3D lifting, M4 CLIP search, M5 benchmark dataset, M6 paper reproduction, M7 ablations, M8 paper/PPT/demo.

## 10. Deliverables

- source code and environment instructions
- dataset protocol and sample manifest
- evaluation scripts and raw result CSV/JSON
- demo screenshots and a recorded fly-through/rotation clip
- failure-case gallery
- IEEE-style paper with related work, method, experiments, results, limitations
- presentation and reproducibility checklist
