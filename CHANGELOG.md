# Changelog

## 3.0.0 — Interactive research workbench

- Reorganized the Streamlit interface into Analyze, Research methods, and Metrics explained tabs.
- Added plain-language explanations for object masks, relative depth, point-cloud limitations, cosine similarity, and latency.
- Added interactive confidence, text-ranking, point-cloud, and per-stage latency charts.
- Added object-table CSV and run-summary JSON downloads.
- Added a research implementation status catalogue for all planned reference-paper tracks.
- Corrected CLIP ranking to use normalized image/text embedding cosine similarity instead of a softmax distribution over crops.
- Cached the YOLO model object after first load to avoid repeated initialization within a session.
- Added unit tests for evaluation helper functions.

### Scope reminder

Version 3 improves usability, transparency, and baseline instrumentation. It does not claim to reproduce all referenced papers. SAM/SAM 2, Splatter Image, LERF, MonoDETR/MonoDTR, VoxFormer, and 4D Gaussian Splatting remain planned method integrations that require their official implementations, data pipelines, checkpoints, and benchmark-specific evaluation.
