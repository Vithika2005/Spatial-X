# Mac Setup and Troubleshooting

## 1. Check Python

Use Python 3.11. Avoid mixing Conda/Miniforge environments and Homebrew Python without checking which executable is active.

```bash
which python
python --version
which pip
```

## 2. Create a clean environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

If OpenCV GUI conflicts occur, keep `opencv-python-headless` rather than installing both OpenCV packages.

## 3. Verify PyTorch/MPS

```bash
python scripts/check_environment.py
```

MPS may be unavailable because of OS/PyTorch compatibility. Consult https://pytorch.org/docs/stable/notes/mps.html. A working app on CPU is preferable to a broken app that assumes Metal support.

## 4. Launch

```bash
streamlit run app.py
```

On first launch, the YOLO weights are downloaded. Depth and CLIP weights download only when those options are enabled. Downloads require network access and consume disk space.

## 5. Common issues

### `No module named ultralytics`
Activate `.venv`, then `pip install -r requirements.txt`.

### Torch/MPS import failure
Confirm Python version and architecture:
```bash
python -c "import platform,torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available())"
```
Reinstall a compatible PyTorch build using official instructions.

### Depth model errors
Uncheck depth for a segmentation-only demo. The depth pipeline is optional and may have Transformers/model compatibility differences. Do not report 3D output if the depth model did not run.

### Slow first launch
Weights are downloaded and cached. First inference includes model startup. Measure repeated inference separately from cold-start time.

### Out of memory / app slow
Lower inference size to 320 or 416, disable CLIP, reduce point cap, and close other GPU-heavy apps.

### `yolov8n-seg.pt` download blocked
Check network/proxy access or download the official Ultralytics model according to its terms. Do not download checkpoints from unknown mirrors.

## 6. Hardware reporting template

Record:
- Mac model and chip
- macOS version
- Python / PyTorch / Ultralytics / Transformers versions
- device actually used by each component
- image dimensions and inference size
- cold-start latency and warm median latency
- whether model download is included
- point cap and CLIP/depth toggle state
