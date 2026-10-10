"""Portable exports and reproducibility metadata for SPATIAL-X."""
from pathlib import Path
from datetime import datetime
import json
import platform
import numpy as np

def software_versions():
    versions={"python":platform.python_version(),"platform":platform.platform()}
    for pkg, modname in [("torch","torch"),("torchvision","torchvision"),("ultralytics","ultralytics"),
                         ("transformers","transformers"),("streamlit","streamlit"),("numpy","numpy"),
                         ("Pillow","PIL"),("plotly","plotly"),("pandas","pandas"),("opencv","cv2")]:
        try:
            mod=__import__(modname); versions[pkg]=getattr(mod,"__version__","installed; version unavailable")
        except Exception: versions[pkg]=None
    try:
        import torch
        versions["mps_available"]=bool(torch.backends.mps.is_available())
        versions["mps_built"]=bool(torch.backends.mps.is_built())
        versions["cuda_available"]=bool(torch.cuda.is_available())
    except Exception: pass
    return versions

def write_ply(path, xyz, rgb, instance_id):
    with open(path,"w",encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n")
        for xyzv, rgbv, iid in zip(xyz,rgb,instance_id):
            f.write(f"{xyzv[0]:.7f} {xyzv[1]:.7f} {xyzv[2]:.7f} {int(rgbv[0])} {int(rgbv[1])} {int(rgbv[2])} {int(iid)}\n")

def save_reproducibility_bundle(result, original_name, points=None, query_scores=None, query_text=None,
                               run_config=None, evaluation_metrics=None):
    import pandas as pd
    out=Path("outputs"); out.mkdir(exist_ok=True)
    stem=Path(original_name).stem.replace(" ","_")
    paths={}
    overlay_path=out/f"{stem}_overlay.png"; result["overlay"].save(overlay_path); paths["overlay"]=overlay_path
    if result.get("depth_vis") is not None:
        depth_path=out/f"{stem}_depth.png"; result["depth_vis"].save(depth_path); paths["depth_visualization"]=depth_path
    if result.get("depth") is not None:
        raw=out/f"{stem}_depth_raw.npy"; np.save(raw,np.asarray(result["depth"],dtype=np.float32)); paths["raw_depth"]=raw
    point_count=0
    if points is not None and len(points.get("xyz",[])):
        xyz=np.asarray(points["xyz"],dtype=np.float32); rgb=np.asarray(points["rgb"],dtype=np.uint8)
        ids=np.asarray(points["instance_id"],dtype=np.int32); point_count=len(xyz)
        ply=out/f"{stem}_pointcloud.ply"; write_ply(ply,xyz,rgb,ids); paths["pointcloud_ply"]=ply
        npz=out/f"{stem}_pointcloud.npz"; np.savez_compressed(npz,xyz=xyz,rgb=rgb,instance_id=ids); paths["pointcloud_npz"]=npz
        csv=out/f"{stem}_pointcloud.csv"
        pd.DataFrame({"x":xyz[:,0],"y":xyz[:,1],"z":xyz[:,2],"red":rgb[:,0],"green":rgb[:,1],"blue":rgb[:,2],"instance_id":ids}).to_csv(csv,index=False)
        paths["pointcloud_csv"]=csv
    query_rows=[]
    if query_scores is not None:
        for inst,score in zip(result.get("instances",[]),query_scores):
            query_rows.append({"query":query_text,"instance_id":inst["instance_id"],"label":inst["label"],
                               "detector_confidence":inst["confidence"],"text_similarity":float(score)})
        query_rows.sort(key=lambda x:x["text_similarity"],reverse=True)
        qcsv=out/f"{stem}_query_scores.csv"; pd.DataFrame(query_rows).to_csv(qcsv,index=False); paths["query_scores_csv"]=qcsv
    metrics=evaluation_metrics or {"status":"not_evaluated","note":"Ground-truth labels and the appropriate benchmark evaluator are required; predictions alone cannot establish accuracy.",
        "metrics":{"detection_map":None,"instance_mask_ap":None,"instance_mask_iou":None,"depth_abs_rel":None,"depth_rmse":None,
                   "depth_delta1":None,"3d_semantic_miou":None,"3d_box_ap":None,"language_recall_at_1":None,"language_recall_at_5":None}}
    instances=[]
    for item in result.get("instances",[]):
        instances.append({k:v for k,v in item.items() if k not in ("mask","crop")})
    metadata={"project":"SPATIAL-X","input_name":original_name,"input_height":int(result["rgb"].shape[0]),
        "input_width":int(result["rgb"].shape[1]),"created_at":datetime.now().astimezone().isoformat(),
        "software_versions":software_versions(),"timings_seconds":result.get("timings",{}),"run_config":run_config or {},
        "query":query_text,"query_scores":query_rows,"num_instances":len(instances),"num_3d_points":point_count,
        "coordinate_warning":"Monocular depth with approximate intrinsics produces pseudo-3D, not metric or globally aligned geometry.",
        "evaluation":metrics,"instances":instances}
    meta=out/f"{stem}_run_metadata.json"; meta.write_text(json.dumps(metadata,indent=2,default=str)); paths["run_metadata_json"]=meta
    evalpath=out/f"{stem}_evaluation_metrics.json"; evalpath.write_text(json.dumps(metrics,indent=2,default=str)); paths["evaluation_metrics_json"]=evalpath
    return paths
