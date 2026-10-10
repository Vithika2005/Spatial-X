from __future__ import annotations

import json
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from spatialx.pipeline import analyze_image, make_point_cloud, encode_query_scores, save_outputs
from spatialx.research_catalog import PAPER_TRACKS, STATUS_LABELS

st.set_page_config(page_title="SPATIAL-X Research Workbench", page_icon="🧭", layout="wide")

# ------------------------------ helpers ------------------------------------
def friendly_seconds(value):
    if value is None:
        return "—"
    return f"{float(value):.2f} s"


def instance_table(instances):
    rows = []
    for item in instances:
        x1, y1, x2, y2 = item["box"]
        rows.append({
            "Object ID": f"Object {item['instance_id']}",
            "Detected class": item["label"],
            "Model confidence": float(item["confidence"]),
            "Box width (px)": int(max(0, x2 - x1)),
            "Box height (px)": int(max(0, y2 - y1)),
            "Mask area (pixels)": int(np.asarray(item["mask"]).sum()),
        })
    return pd.DataFrame(rows)


def method_status_color(status):
    return "success" if status == "implemented_baseline" else "info"


def make_report(result, source_name, config):
    return {
        "project": "SPATIAL-X",
        "source_file": source_name,
        "interpretation": "Single-image estimates; 3D coordinates are pseudo-3D unless calibrated depth and intrinsics are supplied.",
        "configuration": config,
        "timings_seconds": result.get("timings", {}),
        "instances": [
            {"instance_id": int(x["instance_id"]), "label": x["label"],
             "confidence": float(x["confidence"]), "box_xyxy": [int(v) for v in x["box"]]}
            for x in result.get("instances", [])
        ],
        "query": result.get("query_text"),
        "query_cosine_similarities": result.get("query_scores"),
        "evaluation_status": "not_evaluated_without_ground_truth",
    }


st.title("🧭 SPATIAL-X")
st.subheader("From a 2D image to an explorable 3D scene")
st.write("Upload an image, inspect the objects the model finds, explore a depth-based point cloud, and test text queries. Each result is labeled so you can tell a model prediction from a measured fact.")

with st.expander("What does SPATIAL-X do? Start here", expanded=True):
    st.markdown("""
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:stretch">
      <div style="flex:1;min-width:130px;border:1px solid #8885;border-radius:10px;padding:12px"><b>1. Find objects</b><br/>The segmentation model draws a mask around each detected object.</div>
      <div style="font-size:24px;align-self:center">→</div>
      <div style="flex:1;min-width:130px;border:1px solid #8885;border-radius:10px;padding:12px"><b>2. Estimate depth</b><br/>A depth model estimates which pixels appear nearer or farther.</div>
      <div style="font-size:24px;align-self:center">→</div>
      <div style="flex:1;min-width:130px;border:1px solid #8885;border-radius:10px;padding:12px"><b>3. Lift to 3D</b><br/>Pixels are projected into a point cloud using approximate camera geometry.</div>
      <div style="font-size:24px;align-self:center">→</div>
      <div style="flex:1;min-width:130px;border:1px solid #8885;border-radius:10px;padding:12px"><b>4. Search by text</b><br/>Optional CLIP embeddings rank object crops against your phrase.</div>
    </div>
    """, unsafe_allow_html=True)
    st.warning("The 3D view is currently an approximate visualization from one RGB image. It is not a metrically calibrated scan, and the current app does not yet reproduce LERF, Splatter Image, MonoDETR, VoxFormer, or 4D Gaussian Splatting.")

analyze_tab, research_tab, metrics_tab = st.tabs(["🔎 Analyze a scene", "📚 Research methods", "📈 Metrics explained"])

with analyze_tab:
    with st.sidebar:
        st.header("Analysis settings")
        conf = st.slider("Detection threshold", 0.05, 0.90, 0.25, 0.05,
                         help="Raise this to hide weaker detections; lower it if objects are missing.")
        image_size = st.select_slider("Processing resolution", options=[320, 416, 512, 640, 768], value=640,
                                      help="Higher resolution can help small objects but may take longer.")
        max_points = st.select_slider("3D point detail", options=[5000, 10000, 20000, 40000], value=10000,
                                      help="More points make the 3D plot denser and can make it slower.")
        use_depth = st.checkbox("Estimate depth for 3D view", value=True)
        use_clip = st.checkbox("Enable text search (CLIP)", value=False,
                               help="Downloads the CLIP model the first time. It ranks detected crops; it does not segment unseen objects.")
        st.caption("Tip: start with depth on and text search off. First run may download pretrained weights.")

    upload = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
    st.caption("For the clearest demo, use a well-lit indoor photo with several distinct objects.")
    if upload is None:
        st.info("Your workflow: upload a photo → inspect masks → inspect depth → explore 3D → optionally search by text → export results.")
    else:
        try:
            rgb = np.array(Image.open(upload).convert("RGB"))
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")
            st.stop()

        run_config = {"confidence_threshold": conf, "image_size": image_size,
                      "max_points": max_points, "depth_enabled": use_depth, "clip_enabled": use_clip}
        run_start = time.perf_counter()
        with st.spinner("Step 1/3: finding objects and estimating depth..."):
            result = analyze_image(rgb, confidence=conf, image_size=image_size, estimate_depth=use_depth)
        result.setdefault("timings", {})["ui_total_wall_seconds"] = time.perf_counter() - run_start
        instances = result["instances"]
        df = instance_table(instances)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Objects found", len(instances), help="Count of object instances returned by the segmentation model.")
        k2.metric("Image size", f"{rgb.shape[1]} × {rgb.shape[0]}", help="Width × height in pixels.")
        k3.metric("Analysis time", friendly_seconds(result["timings"].get("ui_total_wall_seconds")), help="Includes model loading on a cold first run.")
        k4.metric("Depth map", "Ready" if result.get("depth_vis") is not None else "Unavailable")

        st.markdown("### Step 1 — What objects did the model find?")
        left, right = st.columns([1.25, 1])
        with left:
            st.image(result["overlay"], caption="Each colored region is a predicted object mask. Labels include object ID, class, and model confidence.", use_container_width=True)
        with right:
            if not df.empty:
                conf_fig = px.bar(df.sort_values("Model confidence"), x="Model confidence", y="Object ID", color="Detected class",
                                  orientation="h", range_x=[0, 1], title="How confident is the detector?",
                                  hover_data=["Detected class", "Mask area (pixels)"])
                conf_fig.update_layout(height=max(300, 42 * len(df)), margin=dict(l=5, r=5, t=50, b=5), legend_title="Object type")
                st.plotly_chart(conf_fig, use_container_width=True)
                st.caption("Confidence is the detector's score, not a guarantee that the label is correct.")
            else:
                st.info("No objects were detected. Try a clearer image or lower the detection threshold in the sidebar.")

        if not df.empty:
            with st.expander("See the object table", expanded=False):
                st.dataframe(df.style.format({"Model confidence": "{:.3f}"}), use_container_width=True, hide_index=True)
                st.download_button("Download object table (CSV)", df.to_csv(index=False).encode("utf-8"),
                                   file_name="spatialx_instances.csv", mime="text/csv")

        st.markdown("### Step 2 — How far away might things be?")
        dleft, dright = st.columns([1, 1])
        with dleft:
            if result.get("depth_vis") is not None:
                st.image(result["depth_vis"], caption="Relative depth estimate. Brightness is normalized for this image; it is not distance in metres.", use_container_width=True)
            else:
                st.info("Depth is disabled or the depth model could not load. You can still inspect 2D masks.")
        with dright:
            st.markdown("**How to read this**")
            st.write("The model predicts relative depth from appearance. The display is normalized separately for each image, so its brightness should not be compared numerically across different photos.")
            st.write("To report metric depth error, the prediction must be aligned with ground-truth depth and evaluated using a documented dataset protocol.")

        query_text = None
        query_scores = None
        if use_clip and instances:
            st.markdown("### Step 3 — Find an object using words")
            query_text = st.text_input("Describe the object", placeholder="e.g. a chair, the red bag, a computer monitor")
            if query_text.strip() and st.button("Rank objects for this query", type="primary"):
                with st.spinner("Comparing image crops with the text prompt..."):
                    t0 = time.perf_counter()
                    scores = encode_query_scores(instances, query_text.strip())
                    result["timings"]["clip_query_seconds"] = time.perf_counter() - t0
                if scores is None:
                    st.error("CLIP did not load. Check internet access and installed dependencies, then retry.")
                else:
                    query_scores = [float(x) for x in scores]
                    result["query_scores"] = query_scores
                    result["query_text"] = query_text.strip()
                    ranked = pd.DataFrame({"Object ID": [f"Object {x['instance_id']}" for x in instances],
                                           "Detected class": [x["label"] for x in instances],
                                           "Cosine similarity": query_scores})
                    ranked = ranked.sort_values("Cosine similarity", ascending=False)
                    best = ranked.iloc[0]
                    st.success(f"Top-ranked crop: {best['Object ID']} ({best['Detected class']}). This is the closest embedding match among detected objects.")
                    fig = px.bar(ranked.sort_values("Cosine similarity"), x="Cosine similarity", y="Object ID", color="Detected class",
                                 orientation="h", title=f"Text-to-object ranking: ‘{query_text.strip()}’")
                    fig.update_layout(height=max(300, 42 * len(ranked)), margin=dict(l=5, r=5, t=50, b=5))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Cosine similarity ranks these crops; it is not a probability. This step searches only objects detected by the segmentation model.")

        st.markdown("### Step 4 — Explore the approximate 3D point cloud")
        points = None
        if result.get("depth") is not None:
            with st.spinner("Projecting image pixels into a point cloud..."):
                point_start = time.perf_counter()
                points = make_point_cloud(rgb, result["depth"], instances, max_points=max_points)
                result["timings"]["point_cloud_seconds"] = time.perf_counter() - point_start
        if points is not None and len(points["xyz"]):
            labels_by_id = {0: "Background / unassigned"}
            labels_by_id.update({x["instance_id"]: f"Object {x['instance_id']}: {x['label']}" for x in instances})
            color_mode = st.radio("Color points by", ["Object instance", "Original image color"], horizontal=True)
            cloud = pd.DataFrame({"X": points["xyz"][:, 0], "Y": points["xyz"][:, 1], "Z": points["xyz"][:, 2],
                                  "instance_id": points["instance_id"],
                                  "instance_label": [labels_by_id.get(int(i), "Unknown") for i in points["instance_id"]],
                                  "R": points["rgb"][:, 0], "G": points["rgb"][:, 1], "B": points["rgb"][:, 2]})
            if len(cloud) > 8000:
                cloud = cloud.sample(8000, random_state=7)
            if color_mode == "Object instance":
                fig3d = px.scatter_3d(cloud, x="X", y="Y", z="Z", color="instance_label", opacity=0.78,
                                      title="Each color represents a detected object or background")
            else:
                cloud["RGB"] = [f"rgb({r},{g},{b})" for r, g, b in zip(cloud.R, cloud.G, cloud.B)]
                fig3d = go.Figure(go.Scatter3d(x=cloud.X, y=cloud.Y, z=cloud.Z, mode="markers",
                    marker=dict(size=2, color=cloud.RGB, opacity=0.8), hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>"))
                fig3d.update_layout(title="Points retain their original image colors")
            fig3d.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0), scene=dict(aspectmode="data",
                xaxis_title="X (relative)", yaxis_title="Y (relative)", zaxis_title="Z (pseudo-depth)"))
            st.plotly_chart(fig3d, use_container_width=True)
            st.caption(f"Rendered {len(cloud):,} points for display. The saved point cloud can contain up to {len(points['xyz']):,} points. Rotate and zoom the plot to explore it.")
        else:
            st.info("No point cloud is available. Enable depth estimation and check that the depth model loaded successfully.")

        st.markdown("### Step 5 — What took time?")
        timings = result.get("timings", {})
        timing_labels = {
            "segmentation_seconds": "Find object masks",
            "postprocess_and_overlay_seconds": "Prepare labels and overlay",
            "depth_seconds": "Estimate depth",
            "point_cloud_seconds": "Build 3D points",
            "clip_query_seconds": "Text query (if run)",
            "ui_total_wall_seconds": "Total app run (includes UI/model overhead)",
        }
        tdf = pd.DataFrame([{"Stage": timing_labels.get(k, k.replace("_", " ").title()), "Seconds": float(v)}
                            for k, v in timings.items() if k in timing_labels and isinstance(v, (int, float))])
        if not tdf.empty:
            tf = px.bar(tdf, x="Seconds", y="Stage", orientation="h", title="Time spent in each stage (seconds)")
            tf.update_layout(height=320, margin=dict(l=5, r=5, t=50, b=5))
            st.plotly_chart(tf, use_container_width=True)
            st.caption("The first run may include downloading/loading model weights. Repeat runs are usually a fairer speed comparison.")

        st.markdown("### Step 6 — Save your experiment")
        st.write("The export bundle records outputs, settings, package versions, and timings. Ground-truth evaluation metrics remain unreported until a labeled benchmark is evaluated.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save reproducibility bundle to outputs/", type="primary"):
                paths = save_outputs(result, upload.name, points=points, query_scores=result.get("query_scores"),
                                    query_text=result.get("query_text"), run_config=run_config)
                st.success("Saved experiment artifacts locally.")
                st.json({k: str(v) for k, v in paths.items()})
        with c2:
            report = make_report(result, upload.name, run_config)
            st.download_button("Download run summary (JSON)", json.dumps(report, indent=2).encode("utf-8"),
                               file_name="spatialx_run_summary.json", mime="application/json")

with research_tab:
    st.markdown("## Research status: what is actually implemented?")
    st.write("This table is deliberately honest. A runnable related baseline is not the same as a faithful reproduction of a paper.")
    for track in PAPER_TRACKS:
        with st.expander(f"{track['track']} · {track['method']}", expanded=track["status"] == "implemented_baseline"):
            status = STATUS_LABELS.get(track["status"], track["status"])
            st.markdown(f"**Status:** `{status}`")
            st.markdown(f"**Method / reference:** {track['paper']}")
            st.write(track["what_it_does"])
            st.info("Scope limit: " + track["what_it_does_not"])
            st.markdown(f"**Inputs / data:** {track['data']}")
            st.markdown(f"**Next verification step:** {track['next_validation']}")
    st.markdown("### Recommended implementation order")
    st.markdown("""
    1. **Stabilize the baseline:** test the current YOLO + depth + CLIP flow and save outputs.
    2. **Add benchmark evaluation:** start with a small, permitted ScanNet subset with aligned RGB-D, camera intrinsics, poses, and labels.
    3. **Choose one paper track:** for example, SAM 2 for video masks *or* a KITTI monocular 3D detector. Follow the official repository and report its own benchmark metrics.
    4. **Only then add heavier reconstruction tracks:** Splatter Image, LERF, VoxFormer, and 4D Gaussian Splatting each have distinct data, training, and hardware requirements.
    """)

with metrics_tab:
    st.markdown("## What do the numbers mean?")
    metric_rows = [
        ("Detection confidence", "How strongly the detector scores a proposed object class. It is not guaranteed accuracy."),
        ("Cosine similarity", "How close the CLIP image-crop and text embeddings are in direction. Use it to rank candidates, not as a probability."),
        ("Abs Rel / RMSE", "Depth error metrics that require aligned ground-truth depth and a documented scale/valid-pixel policy."),
        ("Mask IoU / mask AP", "Measures overlap/quality of predicted object masks against ground-truth instance masks."),
        ("Recall@K", "For a labeled text-to-instance query set, the fraction of queries whose correct object appears in the top K results."),
        ("KITTI AP3D / BEV AP", "Official metrics for 3D bounding-box detection. They cannot be computed from the current 2D-only detector output."),
        ("Occupancy mIoU", "Measures voxel semantic occupancy against ground-truth labels. It is not measured by the current point-cloud viewer."),
        ("Latency", "Elapsed time for a named stage. Record hardware, image size, warm/cold run, and repetition policy for fair comparison."),
    ]
    for name, desc in metric_rows:
        st.markdown(f"**{name}**  \n{desc}")
        st.divider()
    st.warning("Do not enter zeros for metrics that have not been evaluated. Use ‘not evaluated’ until the required ground truth and official evaluation protocol are available.")
