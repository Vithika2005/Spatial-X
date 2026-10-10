import platform, sys
try:
    import torch
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch import failed:", repr(e))
    raise
