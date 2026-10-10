# SPATIAL-X — Interactive 2D-to-3D Research Workbench

**A Mac-first deep-learning research project** combining 2D instance segmentation, monocular depth, 3D point-cloud lifting, text-guided object search, and an interactive demo.

> Important scope note: this repository is a runnable research scaffold and baseline implementation, not a claim to have reproduced every component of six independent papers. The runnable core uses pretrained models and geometric lifting. Research-paper reproduction modules such as LERF, Splatter Image, MonoDETR, VoxFormer, and 4D Gaussian Splatting are documented as separate, progressively advanced extensions because their official training/rendering stacks have different hardware and dataset requirements.

## Research goal

Given an RGB image (or sampled video frames), produce:
1. 2D object instances with masks and class labels.
2. An estimated relative depth map.
3. A colored 3D point cloud whose points inherit instance identities.
4. Text-guided ranking of detected object crops.
5. Interactive visualizations and exportable results.

Proposed research title:

**Open-Vocabulary 3D Object Detection and Instance Segmentation from RGB-D and Monocular Images through Cross-Modal Feature Fusion**

Primary public dataset: ScanNet (RGB-D indoor scans and instance annotations). Secondary dataset: KITTI 3D Object Detection for road-scene 3D boxes. Start with your own images/video and the included demo first; use official dataset access terms before downloading benchmark data.

## What works in this repository

- Streamlit UI for image/video upload.
- Ultralytics YOLO segmentation baseline (COCO-pretrained instance masks).
- Optional MiDaS depth estimation through Transformers.
- Camera-intrinsic approximation and RGB-D point-cloud lifting.
- Text query over segmented object crops using CLIP embeddings (optional model download).
- Plotly 3D point-cloud visualization.
- CPU/MPS device selection and basic timing.
- Export overlay, depth image, point cloud CSV, and run metrics.

The outputs are estimates. Monocular depth is generally relative rather than metric depth; point-cloud scale and camera calibration are approximate unless real depth and intrinsics are supplied.

## Quick start (macOS, Apple Silicon)

Use Python 3.11 in a clean environment. Do not use a Python 3.14 environment for this stack.

```bash
git clone <your-repository-url> SPATIAL-X
cd SPATIAL-X
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
streamlit run app.py
```

If `python3.11` is not installed, install it using your preferred package manager (for example Homebrew), then reopen the terminal. PyTorch's MPS backend depends on compatible PyTorch/macOS versions. Check the [Apple PyTorch guide](https://pytorch.org/docs/stable/notes/mps.html).

First launch downloads pretrained model weights and can take time. Internet access is required for first-time model downloads. If an optional model cannot load, use the app's fallback mode and consult `docs/MAC_SETUP.md`.

## Run a quick command-line smoke test

```bash
python scripts/check_environment.py
python scripts/run_image.py --image assets/demo_room.jpg --output outputs
```

The sample image is optional; place any local JPG/PNG at that path or pass your own path. The CLI currently exercises detection/segmentation and saves the annotated image; depth/3D are available through the app.

## System architecture

```text
RGB image / sampled video frames
             |
             v
   YOLO instance segmentation
   boxes + labels + masks
             |
             +--------------------------+
             |                          |
             v                          v
   monocular depth model          CLIP image/text encoder
   relative depth map              rank object crops for prompt
             |                          |
             v                          |
   calibrated/approximate back-projection
             |
             v
   3D points + RGB + instance ID
             |
             v
   Streamlit + Plotly UI / exports
```

## Roadmap for the complete 8A–8F research suite

| Track | Research direction | Role in this project | Feasibility on Mac |
|---|---|---|---|
| 8A | Single-view 3D reconstruction / Gaussian splatting | Extension: novel-view rendering | Research renderers often rely on CUDA; use a supported viewer or remote Linux GPU if required |
| 8B | Dynamic 4D scene reconstruction | Extension: time-varying scene representation | High effort; official methods commonly target NVIDIA CUDA |
| 8C | 2D-to-3D mask propagation | Core geometric extension: project masks through depth/poses | Best next step for this repo |
| 8D | Open-vocabulary 3D language field | Core lightweight version: CLIP-ranked masks lifted to 3D; LERF reproduction is advanced | Lightweight feature ranking can run locally; full LERF training is heavier |
| 8E | Monocular 3D object detection | Separate KITTI track; compare 2D masks and predicted 3D boxes | Official MonoDTR/MonoDETR repos may need CUDA/Linux-specific changes |
| 8F | 3D occupancy / occluded-space completion | Research extension using VoxFormer / SemanticKITTI | Heavy; CUDA-oriented official stack |

Do not claim all six paper methods are reproduced unless you have implemented and evaluated their actual methods. The project is most defensible if the runnable baseline and one or two paper-derived research contributions are completed thoroughly.

## Research contribution you can test

A feasible contribution is **instance-aware cross-modal lifting**:
- Generate 2D instance masks.
- estimate depth;
- back-project valid pixels to 3D;
- attach each 3D point to an instance ID and semantic label;
- compute CLIP similarity between the user's text query and each object crop;
- compare text-guided retrieval with class-label-only retrieval.
- Evaluate the effect of depth quality, mask quality, point sampling, and query ambiguity.

Possible research questions:
1. How does depth quality affect the geometric coherence of lifted instance point clouds?
2. Does CLIP crop ranking improve retrieval of visually similar object instances over detector class labels alone?
3. What latency/quality trade-off results from image resizing and point-cloud subsampling?
4. Which failure modes dominate: occlusion, reflective surfaces, low light, motion blur, or inaccurate camera intrinsics?

## Metrics

Use only metrics appropriate to the available ground truth:
- 2D instance segmentation: mask AP / mask IoU where annotations and evaluation code are available.
- Detection: precision, recall, mAP; avoid comparing results from different splits or protocols.
- Depth: Abs Rel, RMSE, δ thresholds if ground-truth depth is available and aligned.
- 3D semantic segmentation: per-point IoU / mIoU against benchmark labels.
- 3D boxes: official KITTI AP3D / BEV AP with the official evaluation protocol.
- Language retrieval: Recall@K / Precision@K with manually verified query-to-instance labels.
- Systems: end-to-end latency, frames per second, peak process memory (record the measurement method and resolution).

Do not report invented metrics. Save results from actual runs to `outputs/metrics.json`.

## Dataset plan

### ScanNet
Official: https://www.scan-net.org/ and https://github.com/ScanNet/ScanNet  
ScanNet provides RGB-D sequences, camera poses, reconstructed surfaces, and instance-level semantic labels. Access requires accepting the dataset terms. Start with a small permitted subset and document the exact split.

### ScanNet++
Official: https://scannetpp.mlsg.cit.tum.de/scannetpp/  
High-fidelity and very large; the official documentation reports terabyte-scale downloads for full data. Do not download the full dataset just to get the demo working.

### KITTI
Official 3D benchmark: https://www.cvlibs.net/datasets/kitti/eval_3dobject.php  
Use for the monocular 3D detection extension and report the official split/protocol.

### Replica
Official dataset code: https://github.com/facebookresearch/Replica-Dataset  
Useful for indoor reconstruction research; check access and usage terms.

## Paper list (all within the last five years as of October 2026)

1. **LERF: Language Embedded Radiance Fields**, ICCV 2023. Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html  
   Official code: https://github.com/kerrj/lerf  
   Study language embeddings distilled into a 3D radiance field. The app's CLIP-ranked 3D instances are a lightweight related baseline, not a full LERF reproduction.

2. **Splatter Image: Ultra-Fast Single-View 3D Reconstruction**, CVPR 2024. Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Szymanowicz_Splatter_Image_Ultra-Fast_Single-View_3D_Reconstruction_CVPR_2024_paper.html  
   Study feed-forward prediction of 3D Gaussians from a single image. It focuses on object reconstruction and is not itself a full open-vocabulary instance segmentation pipeline.

3. **Segment Anything**, ICCV 2023. Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html  
   Useful for promptable mask generation. SAM/SAM 2 masks do not inherently provide semantic class names; combine them with a detector or language model.

4. **MonoDTR: Monocular 3D Object Detection With Depth-Aware Transformer**, CVPR 2022. Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Huang_MonoDTR_Monocular_3D_Object_Detection_With_Depth-Aware_Transformer_CVPR_2022_paper.html  
   Official code: https://github.com/kuanchihhuang/MonoDTR  
   Relevant to 8E; evaluate installation requirements before choosing it for native Apple Silicon.

5. **MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection**, ICCV 2023. Code: https://github.com/ZrrSkywalker/MonoDETR  
   A useful alternative for KITTI-based monocular 3D detection.

6. **VoxFormer: Sparse Voxel Transformer for Camera-Based 3D Semantic Scene Completion**, CVPR 2023. Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Li_VoxFormer_Sparse_Voxel_Transformer_for_Camera-Based_3D_Semantic_Scene_Completion_CVPR_2023_paper.html  
   Official code: https://github.com/NVlabs/VoxFormer  
   Relevant to 8F. This predicts semantic occupancy/completion, not instance masks alone.

7. **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**, CVPR 2024. Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html  
   Relevant to 8B; the paper's reported RTX 3090 results should not be assumed to apply to an M-series Mac.

8. **Segment Anything 2**, 2024. Project/research: https://ai.meta.com/research/sam2/  
   Useful for propagating masks through video. It can be an optional upgrade to the current instance-segmentation baseline.

9. **ScanNet++**, ICCV 2023. Dataset/documentation: https://scannetpp.mlsg.cit.tum.de/scannetpp/  
   Use only if storage and access are appropriate.

## Suggested project milestones

1. **M1 — Environment and UI:** app starts, upload image, model inference runs.
2. **M2 — 2D baseline:** masks, labels, boxes, annotated export.
3. **M3 — Depth and 3D:** depth map, point-cloud lifting, interactive 3D view.
4. **M4 — Text query:** CLIP ranks object crops and highlights selected instance.
5. **M5 — Dataset evaluation:** create a reproducible small validation set and metrics script.
6. **M6 — Paper module:** implement or adapt one research-paper method; record changes and compare to baseline.
7. **M7 — Ablations:** depth/no depth, CLIP/no CLIP, resolution and point-sampling tests.
8. **M8 — Final demo and paper:** failure cases, results tables, presentation, limitations, citations.

## Important hardware reality

Apple Silicon supports PyTorch MPS for many operations, but that does not make CUDA-only custom kernels compatible with Metal. Neural rendering and 3D Gaussian Splatting repositories often use custom CUDA rasterizers. Check each official repository's current setup instructions before promising native execution. This project's baseline is designed to be Mac-first; some paper reproductions may require a compatible Linux/NVIDIA machine or a hosted GPU.

## License and academic integrity

Follow the code license for every external project and the terms for every dataset/model checkpoint. Cite the original papers. Distinguish pretrained inference, code adaptation, reproduced paper components, and your own contribution in the report.


## Reproducibility exports (updated)

Click **Save reproducibility bundle to outputs/**. Exports include the point cloud as PLY/NPZ/CSV, raw depth, CLIP query scores when used, per-stage timings, package versions, run configuration, and evaluation metrics. Metrics requiring ground truth remain null with status `not_evaluated` until the appropriate benchmark evaluator is run. See `docs/EXPORTS_AND_EVALUATION.md`.


## Version 3 — Beginner-friendly research workbench

Version 3 improves the app's explanations and visualizations. It includes confidence charts, object tables, a clearer relative-depth explanation, point-cloud coloring by instance or source RGB, per-stage latency charts, downloadable run summaries, and a research tracker that distinguishes the runnable baselines from paper methods that are not yet implemented. CLIP text matching uses cosine similarity for ranking; these scores are not probabilities.

### Launch

```bash
source .venv/bin/activate
streamlit run app.py
```

### What is implemented vs. planned?

| Component | Status | Honest interpretation |
|---|---|---|
| YOLOv8-seg | Runnable pretrained baseline | 2D masks and labels; not SAM/SAM 2 |
| MiDaS via Transformers | Optional pretrained baseline | Relative depth estimate; not guaranteed metric depth |
| Point-cloud lifting | Implemented geometry visualization | Approximate coordinates with guessed intrinsics; not a calibrated reconstruction |
| CLIP ViT-B/32 | Optional text/crop retrieval baseline | Ranks detected crops by cosine similarity; not LERF |
| Splatter Image | Planned | Not implemented/reproduced in this repository |
| LERF | Planned | Not implemented/reproduced in this repository |
| MonoDETR / MonoDTR | Planned | Not producing 3D boxes yet |
| VoxFormer | Planned | No semantic occupancy prediction yet |
| 4D Gaussian Splatting | Planned | Video frame selection is not 4D reconstruction |
| SAM / SAM 2 | Planned | Current masks come from YOLOv8-seg |

See `docs/RESEARCH_IMPLEMENTATION_STATUS.md` before describing the project in a poster or paper. The reference papers are different methods with different inputs, training code, datasets, and evaluation protocols. They cannot be truthfully described as all implemented merely because the app has a 3D plot.
