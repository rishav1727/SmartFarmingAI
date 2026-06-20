import torch
import time
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    a = torch.randn(1000, 1000).to("cuda")
    b = torch.randn(1000, 1000).to("cuda")
    print("Testing matrix multiplication...")
    for i in range(10):
        c = torch.matmul(a, b)
        print(f"Iter {i} done")
    print("Success!")
else:
    print("CUDA not available")
