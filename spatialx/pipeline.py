from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch

_DEPTH_PIPE = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_DETECTOR = None

def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from ultralytics import YOLO
        # COCO-pretrained instance segmentation baseline; weights download on first run.
        _DETECTOR = YOLO("yolov8n-seg.pt")
    return _DETECTOR

def _load_depth():
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        # Transformers' generic pipeline does not consistently route to MPS across all versions.
        # CPU is the conservative fallback. Change to a tested MPS-specific implementation if desired.
        _DEPTH_PIPE = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=device)
    return _DEPTH_PIPE

def _overlay(rgb, instances):
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0,0,0,0))
    pix = np.array(layer)
    palette = [(255,70,70),(50,220,120),(50,150,255),(255,190,40),(190,80,255),(30,220,220)]
    for j, item in enumerate(instances):
        mask = item["mask"]
        color = palette[j % len(palette)]
        pix[mask, :3] = color
        pix[mask, 3] = 85
    layer = Image.fromarray(pix, "RGBA")
    image = Image.alpha_composite(image, layer)
    draw = ImageDraw.Draw(image)
    for item in instances:
        x1,y1,x2,y2 = item["box"]
        draw.rectangle((x1,y1,x2,y2), outline=(255,255,0,255), width=2)
        draw.text((x1, max(0,y1-16)), f'{item["instance_id"]}: {item["label"]} {item["confidence"]:.2f}', fill=(255,255,0,255))
    return image.convert("RGB")

def analyze_image(rgb, confidence=0.25, image_size=640, estimate_depth=True):
    rgb = np.asarray(rgb).astype(np.uint8)
    total_started = time.perf_counter()
    timings = {}
    stage_started = time.perf_counter()
    model = _load_detector()
    pred = model.predict(source=rgb, conf=confidence, imgsz=image_size, verbose=False)[0]
    timings["segmentation_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    instances = []
    if pred.boxes is not None and len(pred.boxes):
        boxes = pred.boxes.xyxy.detach().cpu().numpy()
        confs = pred.boxes.conf.detach().cpu().numpy()
        clss = pred.boxes.cls.detach().cpu().numpy().astype(int)
        masks = pred.masks.data.detach().cpu().numpy() if pred.masks is not None else []
        names = pred.names
        h,w = rgb.shape[:2]
        for i, (box, score, cls) in enumerate(zip(boxes, confs, clss)):
            if len(masks):
                # Ultralytics mask arrays are usually resized to original image dimensions.
                m = masks[i]
                if m.shape != (h,w):
                    m = np.array(Image.fromarray((m*255).astype(np.uint8)).resize((w,h))) > 127
                else:
                    m = m > 0.5
            else:
                m = np.zeros((h,w), dtype=bool)
                x1,y1,x2,y2 = np.clip(box.astype(int), [0,0,0,0], [w-1,h-1,w-1,h-1])
                m[y1:y2+1,x1:x2+1] = True
            x1,y1,x2,y2 = [int(v) for v in box]
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            crop = Image.fromarray(rgb[y1:y2, x1:x2]) if x2>x1 and y2>y1 else Image.fromarray(rgb)
            instances.append({"instance_id":i+1, "label":str(names[int(cls)]), "class_id":int(cls),
                "confidence":float(score), "box":[x1,y1,x2,y2], "mask":m, "crop":crop})
    overlay = _overlay(rgb, instances)
    timings["postprocess_and_overlay_seconds"] = time.perf_counter() - stage_started
    depth = depth_vis = None
    stage_started = time.perf_counter()
    if estimate_depth:
        try:
            depth_pipe = _load_depth()
            depth_result = depth_pipe(Image.fromarray(rgb))
            d = depth_result["predicted_depth"]
            depth = np.asarray(d.squeeze().detach().cpu() if hasattr(d,"detach") else d, dtype=np.float32)
            if depth.shape != rgb.shape[:2]:
                depth = np.array(Image.fromarray(depth).resize((rgb.shape[1],rgb.shape[0])))
            # Normalize for visualization only. Keep raw model values for lifting.
            lo, hi = np.percentile(depth, [2,98])
            dn = np.clip((depth-lo)/(hi-lo+1e-8),0,1)
            depth_vis = Image.fromarray((dn*255).astype(np.uint8)).convert("RGB")
        except Exception as exc:
            # Keep the segmentation app useful even if optional depth model fails.
            depth = depth_vis = None
            timings["depth_error"] = repr(exc)
    timings["depth_seconds"] = time.perf_counter() - stage_started
    timings["total_pipeline_seconds"] = time.perf_counter() - total_started
    return {"rgb":rgb, "overlay":overlay, "instances":instances, "depth":depth,
            "depth_vis":depth_vis, "timings":timings, "inference_seconds":timings["total_pipeline_seconds"],
            "query_scores":None, "query_text":None}

def make_point_cloud(rgb, depth, instances, max_points=20000):
    if depth is None:
        return None
    rgb = np.asarray(rgb)
    h,w = rgb.shape[:2]
    d = np.asarray(depth, dtype=np.float32)
    if d.shape != (h,w):
        d = np.array(Image.fromarray(d).resize((w,h)))
    # Relative-depth models may invert depth convention; this is a visualization projection,
    # not a calibrated metric reconstruction. Normalize to positive pseudo-depth.
    finite = np.isfinite(d)
    lo, hi = np.percentile(d[finite], [2,98]) if finite.any() else (0,1)
    z = np.clip((d-lo)/(hi-lo+1e-8),0.03,1.0)
    z = 1.0 / z
    fx = fy = 0.9 * max(h,w)  # approximate focal length; replace with calibrated intrinsics for research
    cx,cy = w/2.0,h/2.0
    yy,xx = np.indices((h,w))
    X = (xx-cx)*z/fx
    Y = -(yy-cy)*z/fy
    ids = np.zeros((h,w), dtype=np.int32)
    for item in instances:
        ids[item["mask"]] = item["instance_id"]
    valid = finite & (z < np.percentile(z[finite], 99.5) if finite.any() else False)
    flat = np.flatnonzero(valid)
    if len(flat) > max_points:
        rng = np.random.default_rng(42)
        flat = np.sort(rng.choice(flat, size=max_points, replace=False))
    return {"xyz":np.column_stack((X.ravel()[flat],Y.ravel()[flat],z.ravel()[flat])),
            "rgb":rgb.reshape(-1,3)[flat], "instance_id":ids.ravel()[flat]}

def encode_query_scores(instances, query):
    """Return cosine similarities between a text prompt and detected object crops.

    These are CLIP embedding similarities, not calibrated probabilities. Values are
    useful for ranking candidates from the same image, not as confidence estimates.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR
    try:
        from transformers import CLIPModel, CLIPProcessor
        if _CLIP_MODEL is None:
            model_id = "openai/clip-vit-base-patch32"
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
            _CLIP_MODEL = CLIPModel.from_pretrained(model_id)
            _CLIP_MODEL.eval()
        crops = [x["crop"] for x in instances]
        image_inputs = _CLIP_PROCESSOR(images=crops, return_tensors="pt", padding=True)
        text_inputs = _CLIP_PROCESSOR(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = _CLIP_MODEL.get_image_features(**image_inputs)
            text_features = _CLIP_MODEL.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = (image_features @ text_features.T).squeeze(-1)
        return scores.cpu().numpy()
    except Exception:
        return None

def save_outputs(result, original_name, points=None, query_scores=None, query_text=None, run_config=None, evaluation_metrics=None):
    from spatialx.reproducibility import save_reproducibility_bundle
    return save_reproducibility_bundle(result, original_name, points=points,
        query_scores=query_scores, query_text=query_text, run_config=run_config,
        evaluation_metrics=evaluation_metrics)
