import time
import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("No GPU detected by torch.")
        return

    print("CUDA device count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

    # Some builds expose HIP version for ROCm
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.version.hip:", getattr(torch.version, "hip", None))

    device = torch.device("cuda")

    # Simple correctness check
    x = torch.randn(2000, 2000, device=device)
    y = torch.randn(2000, 2000, device=device)
    z = x @ y
    torch.cuda.synchronize()

    print("Tensor device:", z.device)
    print("Result shape:", z.shape)
    print("Result mean:", z.mean().item())

    # Simple speed check
    n = 4096
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # Warmup
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / 10 * 1000
    print(f"Average {n}x{n} matmul time: {avg_ms:.2f} ms")

if __name__ == "__main__":
    main()