import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# matrix size (increase if you want)
N = 4096

# warm-up (important for fair timing)
a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)
_ = a @ b
torch.cuda.synchronize()

# timed run
start = time.time()

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()  # wait for GPU to finish
end = time.time()

print(f"Time for 10 matmuls: {end - start:.4f} seconds")
print(f"Average per matmul: {(end - start)/10:.4f} seconds")
