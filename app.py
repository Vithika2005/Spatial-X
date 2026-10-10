from __future__ import annotations
import io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from spatialx.pipeline import analyze_image, make_point_cloud, encode_query_scores, save_outputs

st.set_page_config(page_title="SPATIAL-X", page_icon="🧭", layout="wide")
st.title("🧭 SPATIAL-X")
st.caption("Open-vocabulary 3D object detection & instance segmentation — research demo")

with st.sidebar:
    st.header("Pipeline settings")
    conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768], value=640)
    max_points = st.select_slider("Maximum 3D points", options=[5000, 10000, 20000, 40000], value=20000)
    use_depth = st.checkbox("Estimate monocular depth (first run downloads model)", value=True)
    use_clip = st.checkbox("Enable CLIP text query (first run downloads model)", value=False)
    st.caption("For quick setup, first run with CLIP disabled. Models are downloaded from their publishers.")

upload = st.file_uploader("Upload an image (JPG/PNG) or video (MP4/MOV)", type=["jpg","jpeg","png","mp4","mov","m4v"])
if upload is None:
    st.info("Upload a room image to see 2D instance masks, estimated depth, and a 3D point cloud.")
    st.markdown("**Recommended demo:** take a well-lit photo of a room with distinct chairs, tables, monitors, and objects. Avoid private or sensitive imagery.")
    st.stop()

is_video = upload.type.startswith("video") or Path(upload.name).suffix.lower() in {".mp4",".mov",".m4v"}
frames = []
if is_video:
    import cv2
    tmp = Path("outputs"); tmp.mkdir(exist_ok=True)
    vp = tmp / "uploaded_video" + Path(upload.name).suffix if False else tmp / ("uploaded_video" + Path(upload.name).suffix)
    vp.write_bytes(upload.getvalue())
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_frame = max(0, total - 1)
    frame_idx = st.slider("Choose frame", 0, max_frame, min(max_frame, total // 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        st.error("Could not read this video frame. Try MP4 encoded with H.264.")
        st.stop()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
else:
    rgb = np.array(Image.open(upload).convert("RGB"))

start = time.perf_counter()
with st.spinner("Running instance segmentation and optional spatial models..."):
    result = analyze_image(rgb, confidence=conf, image_size=image_size, estimate_depth=use_depth)
elapsed = time.perf_counter() - start
result.setdefault("timings", {})["ui_total_wall_seconds"] = elapsed

col1, col2 = st.columns(2)
with col1:
    st.subheader("2D instance segmentation")
    st.image(result["overlay"], use_container_width=True)
with col2:
    st.subheader("Estimated depth")
    if result.get("depth_vis") is not None:
        st.image(result["depth_vis"], caption="Relative depth visualization; not metric distance", use_container_width=True)
    else:
        st.warning("Depth estimation disabled.")

st.metric("End-to-end inference time", f"{elapsed:.2f} s")
instances = result["instances"]
if instances:
    df = pd.DataFrame([{k:v for k,v in item.items() if k not in ("mask", "crop")} for item in instances])
    st.subheader("Detected instances")
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.warning("No instances found at this confidence threshold. Lower confidence or try a clearer image.")

if use_clip and instances:
    query = st.text_input("Text query", placeholder="e.g. chair, red bag, monitor")
    if query.strip():
        clip_start = time.perf_counter()
        with st.spinner("Ranking detected object crops with CLIP..."):
            scores = encode_query_scores(instances, query.strip())
        result["timings"]["clip_query_seconds"] = time.perf_counter() - clip_start
        if scores is not None:
            result["query_scores"] = [float(v) for v in scores]
            result["query_text"] = query.strip()
            for inst, score in zip(instances, scores):
                inst["text_score"] = float(score)
            best = int(np.argmax(scores))
            st.success(f"Best matching detected instance: {instances[best]['label']} (similarity {scores[best]:.3f})")
            st.dataframe(pd.DataFrame([{"instance_id":x["instance_id"],"label":x["label"],"text_similarity":round(x["text_score"],4)} for x in instances]).sort_values("text_similarity", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("CLIP could not load. Check network access and dependencies.")

st.subheader("3D point cloud")
points = None
if result.get("depth") is not None:
    point_start = time.perf_counter()
    points = make_point_cloud(rgb, result["depth"], instances, max_points=max_points)
    result["timings"]["point_cloud_seconds"] = time.perf_counter() - point_start
    if points is not None and len(points["xyz"]):
        cloud = pd.DataFrame({"X":points["xyz"][:,0], "Y":points["xyz"][:,1], "Z":points["xyz"][:,2],
                              "R":points["rgb"][:,0], "G":points["rgb"][:,1], "B":points["rgb"][:,2],
                              "instance_id":points["instance_id"]})
        # Plotly 3D scatter uses RGB strings to preserve source colors.
        cloud["color"] = [f"rgb({r},{g},{b})" for r,g,b in points["rgb"]]
        fig = go.Figure(data=[go.Scatter3d(x=cloud.X, y=cloud.Y, z=cloud.Z, mode="markers",
            marker=dict(size=2, color=cloud.color, opacity=0.8), text=["instance "+str(i) for i in cloud.instance_id])])
        fig.update_layout(height=650, margin=dict(l=0,r=0,t=10,b=0), scene=dict(aspectmode="data",
            xaxis_title="X (relative)", yaxis_title="Y (relative)", zaxis_title="Z (relative)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This point cloud is lifted from monocular relative depth with approximate intrinsics. It is not a metrically calibrated reconstruction.")
    else:
        st.info("No valid points to render. Try another image or adjust the image.")
else:
    st.info("Enable depth estimation to construct the 3D point cloud.")

st.subheader("Reproducibility record")
with st.expander("Per-stage latency and configuration", expanded=False):
    st.json({"timings_seconds": result.get("timings", {}),
             "configuration": {"confidence": conf, "image_size": image_size, "max_points": max_points,
                               "use_depth": use_depth, "use_clip": use_clip}})

if st.button("Save reproducibility bundle to outputs/"):
    paths = save_outputs(result, upload.name, points=points,
        query_scores=result.get("query_scores"), query_text=result.get("query_text"),
        run_config={"confidence": conf, "image_size": image_size, "max_points": max_points,
                    "use_depth": use_depth, "use_clip": use_clip})
    st.success("Saved reproducibility artifacts.")
    st.json({k:str(v) for k,v in paths.items()})
