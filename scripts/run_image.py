import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from spatialx.pipeline import analyze_image, save_outputs

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--output", default="outputs")
p.add_argument("--confidence", type=float, default=0.25)
args = p.parse_args()
img = np.array(Image.open(args.image).convert("RGB"))
result = analyze_image(img, confidence=args.confidence, estimate_depth=False)
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
target = out / (Path(args.image).stem + "_overlay.png")
result["overlay"].save(target)
print("Saved:", target)
print("Instances:", len(result["instances"]))
for x in result["instances"]:
    print(x["instance_id"], x["label"], round(x["confidence"], 3), x["box"])
