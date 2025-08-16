import torch
import time
from icecream import ic
import packbits


def gen_input(b, h, n, d, d_f, max_chunk_bits):
    device = "cuda"

    q = torch.randn(b, h, n, d, device=device)
    k = torch.randn(b, h, n, d, device=device)
    v = torch.randn(b, h, n, d, device=device)

    # f construction (unchanged)
    f = torch.zeros(n * d_f, device=device, dtype=torch.float32)
    perm = torch.randperm(n * d_f, device=device)[: int(n * d_f * 0.05)]
    f[perm] = 1.0
    f = f.reshape(1, 1, n, d_f)

    # Convert to binary int32
    f_binary = (f > 0).squeeze(0).squeeze(0).to(torch.int32)  # (n, d_f)

    weights = 2 ** torch.arange(max_chunk_bits, device=device, dtype=torch.int32)

    NUM_REP = 10000
    # Compute all chunks in one go
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10000):
        chunks = []
        for start in range(0, d_f, max_chunk_bits):
            end = min(start + max_chunk_bits, d_f)
            bits_chunk = f_binary[:, start:end]  # (n, chunk_size)
            w = weights[: end - start]  # (chunk_size,)
            packed_chunk = (bits_chunk * w).sum(dim=1, dtype=torch.int32)  # (n,)
            chunks.append(packed_chunk)

        fint = torch.stack(chunks, dim=1).reshape(-1)  # shape (n*#chunks,)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    ic(elapsed)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10000):
        fint_kernel = packbits.pack_bits(f_binary, max_chunk_bits)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    ic(elapsed)

    diff = (fint - fint_kernel).abs().float().mean().item()
    ic(diff)


b = 1
h = 8
n = 50130
d = 192
d_f = 151
max_chunk_bits = 25
gen_input(b, h, n, d, d_f, max_chunk_bits)
