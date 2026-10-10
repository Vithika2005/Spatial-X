# Research Paper and PPT Blueprint

## Paper structure (IEEE conference format)

1. Title and abstract: define the problem, method, dataset, and measured results only after experiments.
2. Introduction: gap between 2D instance masks, spatial representation, and language-based object retrieval.
3. Related work:
   - 2D instance segmentation and promptable segmentation (SAM/SAM 2)
   - monocular depth and RGB-D reconstruction
   - language-grounded 3D representations (LERF)
   - Gaussian splatting and single-view reconstruction (Splatter Image)
   - monocular 3D boxes (MonoDTR/MonoDETR)
   - occupancy completion (VoxFormer)
4. Method:
   - input normalization
   - detector and mask extraction
   - depth estimation
   - camera projection
   - instance-aware point cloud
   - language query ranking
   - confidence filtering (if implemented as contribution)
5. Experimental setup: dataset access, split, hardware, software, pretrained weights, image size.
6. Results: measured metrics with mean/median and uncertainty.
7. Ablation study: component-wise comparison.
8. Failure cases and limitations.
9. Conclusion and future work.
10. References and reproducibility appendix.

## Avoid these claims unless proved
- “real-time” without measured FPS and stated hardware.
- “metric 3D reconstruction” from uncalibrated monocular relative depth.
- “reproduced LERF” when only CLIP crop ranking is implemented.
- “M5 Neural Engine acceleration” unless the actual model runs through a verified Core ML path.
- “outperforms state of the art” without the same dataset, split, metric, and official protocol.

## PPT outline (12–14 slides)

1. Title: SPATIAL-X.
2. Motivation: 2D pixels versus 3D object-level understanding.
3. Problem statement and research questions.
4. Related work timeline (2022–2024 papers).
5. Dataset: ScanNet and optional KITTI.
6. System architecture diagram.
7. Detection and instance segmentation results.
8. Depth estimation and geometric lifting.
9. Interactive 3D point cloud demo.
10. Text-guided retrieval demo.
11. Quantitative results and latency.
12. Ablation study.
13. Failure cases and limitations.
14. Conclusion, contribution, references.

## Live demo script (3 minutes)

1. Upload a room image.
2. Show detected instances and colored masks.
3. Enable depth and explain relative depth.
4. Rotate the point cloud and explain the approximate camera projection.
5. Enable CLIP and query “chair” or “monitor”.
6. Change confidence threshold and show a false positive or missed object.
7. Show output artifacts and metrics.
8. End with one failure case: reflective glass, occlusion, or a small object.

## Suggested figures
- Architecture diagram.
- Original image / mask overlay / depth map / point cloud panel.
- Query ranking table.
- Latency-versus-resolution plot.
- Ground-truth versus predicted depth (only when aligned ground truth exists).
- Failure gallery with captions.
