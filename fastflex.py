import time
from tqdm import trange
from icecream import ic
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

flex_attention = torch.compile(flex_attention)


def scaled_dot_product_attention_with_flags(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    flag: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Scaled Dot-Product Attention with a flag-based mask.

    Args:
        query (torch.Tensor): Query tensor of shape (..., Seq_len, E).
        key (torch.Tensor): Key tensor of shape (..., Seq_len, E).
        value (torch.Tensor): Value tensor of shape (..., Seq_len, E).
        flag (torch.Tensor): Flag tensor of shape (..., Seq_len, F).
                             Elements must be 0 or 1.

    Returns:
        torch.Tensor: The output of the attention operation.
    """
    # Generate the custom mask based on the flag tensor.
    # We compute the dot product of all pairs of flag vectors.
    flag_dot_products = torch.matmul(flag, flag.transpose(-2, -1))

    # Create a boolean mask: `True` where the dot product is non-zero.
    flag_mask = flag_dot_products > 0.0

    # ic(flag_mask.shape)

    # Use the PyTorch scaled_dot_product_attention API.
    # The API handles the scaling, softmax, dropout, and matmul with value.
    # We pass our flag_mask to the `attn_mask` argument.
    output = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=flag_mask,
        dropout_p=0.0,
        is_causal=False,
    )

    return output


def gen_input(b, h, n, d, d_f):
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

    # Prepare weights for up to 30 bits at a time
    max_chunk_bits = 30
    weights = 2 ** torch.arange(max_chunk_bits, device=device, dtype=torch.int32)

    # Compute all chunks in one go
    chunks = []
    for start in range(0, d_f, max_chunk_bits):
        end = min(start + max_chunk_bits, d_f)
        bits_chunk = f_binary[:, start:end]  # (n, chunk_size)
        w = weights[: end - start]  # (chunk_size,)
        packed_chunk = (bits_chunk * w).sum(dim=1, dtype=torch.int32)  # (n,)
        chunks.append(packed_chunk)

    fint = torch.stack(chunks, dim=1).reshape(-1)  # shape (n*#chunks,)

    return q, k, v, f, fint


b = 1
h = 8
n = 5013
d = 128
d_f = 151


q, k, v, f, fint = gen_input(b, h, n, d, d_f)


def dense_mod(score, b, h, q_idx, kv_idx):
    part_0 = fint[q_idx * 6 + 0].bitwise_and(fint[kv_idx * 6 + 0])
    part_1 = fint[q_idx * 6 + 1].bitwise_and(fint[kv_idx * 6 + 1])
    part_2 = fint[q_idx * 6 + 2].bitwise_and(fint[kv_idx * 6 + 2])
    part_3 = fint[q_idx * 6 + 3].bitwise_and(fint[kv_idx * 6 + 3])
    part_4 = fint[q_idx * 6 + 4].bitwise_and(fint[kv_idx * 6 + 4])
    part_5 = fint[q_idx * 6 + 5].bitwise_and(fint[kv_idx * 6 + 5])
    sum = part_0 + part_1 + part_2 + part_3 + part_4 + part_5
    return torch.where(sum != 0, score, -float("inf"))


# do dummy matmul in a loop to warmup
for _ in range(100):
    _ = q.matmul(k.transpose(-2, -1))
torch.cuda.synchronize()

N_REP = 10
torch.cuda.synchronize()
start = time.time()
for _ in trange(N_REP):
    ref = scaled_dot_product_attention_with_flags(
        q,
        k,
        v,
        f,
    )
torch.cuda.synchronize()
stop = time.time()
print("PyTorch (ref):", stop - start)

kernel_options = {
    "BLOCK_M": 16,
    "BLOCK_N": 16,
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}
for _ in trange(3):
    out = flex_attention(
        q,
        k,
        v,
        score_mod=dense_mod,
        kernel_options=kernel_options,
    )

torch.cuda.synchronize()
start = time.time()
for _ in trange(N_REP):
    out = flex_attention(
        q,
        k,
        v,
        score_mod=dense_mod,
        kernel_options=kernel_options,
    )
torch.cuda.synchronize()
stop = time.time()
print("flex:", stop - start)

diff = (ref - out).abs()
diffmean = diff.mean()
diffmax = diff.max()
idx = diff.abs().argmax()

ic(ref.shape)
ic(out.shape)
ic(idx)
ic(diff.flatten()[idx - 10 : idx + 10])
ic(diffmean)
ic(diffmax)
