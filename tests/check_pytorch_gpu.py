import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

assert torch.cuda.is_available(), "PyTorch cannot access the NVIDIA GPU"

device = torch.device("cuda")
x = torch.randn(4096, 4096, device=device)
y = x @ x

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute device: {y.device}")
print(f"Result shape: {tuple(y.shape)}")
