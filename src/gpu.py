import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)  # None => CPU-only torch
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
