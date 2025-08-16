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

    # Create a boolean mask: `True` where the dot product is zero.
    # These are the positions where attention scores should be zeroed out.
    flag_mask = flag_dot_products > 0.0

    ic(flag_mask.shape)

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


b = 1
h = 1
n = 1320
d = 128
d_f = 151

device = "cuda"

q = torch.randn(b, h, n, d, device=device)
k = torch.randn(b, h, n, d, device=device)
v = torch.randn(b, h, n, d, device=device)
f = torch.zeros(n * d_f, device=device, dtype=torch.float32)
# fill 5% of f with ones
perm = torch.randperm(n * d_f)[: int(n * d_f * 0.05)]
f[perm] = 1.0
f = f.reshape(1, 1, n, d_f)

# 5 * int32 has 160 bits which is enough to encode 151 0-or-1 values.
# pad the remaining with zeros.
f0 = torch.zeros(n, dtype=torch.int32, device=device)
f1 = torch.zeros(n, dtype=torch.int32, device=device)
f2 = torch.zeros(n, dtype=torch.int32, device=device)
f3 = torch.zeros(n, dtype=torch.int32, device=device)
f4 = torch.zeros(n, dtype=torch.int32, device=device)
f5 = torch.zeros(n, dtype=torch.int32, device=device)
# TODO: fill in fi with content in f.
# first 2 dims are squeezed. this is NOT A TYPO. the first 2 dims are 1, 1.
# just treat int32 as raw bits. Only use the low 30 bits to avoid overflow and sign bit.


# Pack binary values into int32 tensors (treating as signed)
f_binary = (f > 0).squeeze(0).squeeze(0).to(torch.int32)  # Shape: (n, d_f)

for i in range(n):
    # Extract the 151 binary values for this sequence position
    bits = f_binary[i]  # Shape: (151,) /////.

    # Pack into 5 int32 values, using only 31 bits per int32 (avoid sign bit)
    # f0: bits 0-30 (31 bits)
    if len(bits) > 0:
        end_idx = min(30, len(bits))
        f0[i] = torch.sum(bits[:end_idx] * (2 ** torch.arange(end_idx, device=device)))

    # f1: bits 31-61 (31 bits)
    if len(bits) > 30:
        start_idx = 30
        end_idx = min(60, len(bits))
        f1[i] = torch.sum(bits[start_idx:end_idx] * (2 ** torch.arange(end_idx - start_idx, device=device)))

    # f2: bits 62-92 (31 bits)
    if len(bits) > 60:
        start_idx = 60
        end_idx = min(90, len(bits))
        f2[i] = torch.sum(bits[start_idx:end_idx] * (2 ** torch.arange(end_idx - start_idx, device=device)))

    # f3: bits 93-123 (31 bits)
    if len(bits) > 90:
        start_idx = 90
        end_idx = min(120, len(bits))
        f3[i] = torch.sum(bits[start_idx:end_idx] * (2 ** torch.arange(end_idx - start_idx, device=device)))

    # f4: bits 124-150 (27 bits, well within 31-bit limit)
    if len(bits) > 120:
        start_idx = 120
        end_idx = min(150, len(bits))
        f4[i] = torch.sum(bits[start_idx:end_idx] * (2 ** torch.arange(end_idx - start_idx, device=device)))

    if len(bits) > 150:
        start_idx = 150
        end_idx = min(d_f, len(bits))
        f5[i] = torch.sum(bits[start_idx:end_idx] * (2 ** torch.arange(end_idx - start_idx, device=device)))


# Verification: unpack and check
def unpack_bits_signed(f0, f1, f2, f3, f4, n, d_f):
    """Unpack the signed int32 tensors back to binary for verification"""
    unpacked = torch.zeros(n, d_f, dtype=torch.int32, device=device)

    for i in range(n):
        bit_idx = 0

        # Unpack f0 (bits 0-30, 31 bits)
        val = f0[i].item()
        for j in range(30):
            if bit_idx < d_f:
                unpacked[i, bit_idx] = (val >> j) & 1
                bit_idx += 1

        # Unpack f1 (bits 31-61, 31 bits)
        val = f1[i].item()
        for j in range(30):
            if bit_idx < d_f:
                unpacked[i, bit_idx] = (val >> j) & 1
                bit_idx += 1

        # Unpack f2 (bits 62-92, 31 bits)
        val = f2[i].item()
        for j in range(30):
            if bit_idx < d_f:
                unpacked[i, bit_idx] = (val >> j) & 1
                bit_idx += 1

        # Unpack f3 (bits 93-123, 31 bits)
        val = f3[i].item()
        for j in range(30):
            if bit_idx < d_f:
                unpacked[i, bit_idx] = (val >> j) & 1
                bit_idx += 1

        # Unpack f4 (bits 124-150, 27 bits)
        val = f4[i].item()
        for j in range(30):
            if bit_idx < d_f:
                unpacked[i, bit_idx] = (val >> j) & 1
                bit_idx += 1

        # Unpack f5 (bits 151-d_f, as many bits as needed)
        val = f5[i].item()
        for j in range(d_f - bit_idx):
            if bit_idx < d_f:
                unpacked[i, bit_idx] = (val >> j) & 1
                bit_idx += 1

    return unpacked


# Verify the packing worked correctly
unpacked = unpack_bits_signed(f0, f1, f2, f3, f4, n, d_f)
original_binary = f.squeeze(0).squeeze(0).to(torch.int32)
print("Packing verification:", torch.equal(unpacked, original_binary))

# Additional check: ensure no negative values (sign bit not set)
print("f0 non-negative:", torch.all(f0 >= 0))
print("f1 non-negative:", torch.all(f1 >= 0))
print("f2 non-negative:", torch.all(f2 >= 0))
print("f3 non-negative:", torch.all(f3 >= 0))
print("f4 non-negative:", torch.all(f4 >= 0))


def dense_mod(score, b, h, q_idx, kv_idx):
    part_0 = f0[q_idx].bitwise_and(f0[kv_idx])
    part_1 = f1[q_idx].bitwise_and(f1[kv_idx])
    part_2 = f2[q_idx].bitwise_and(f2[kv_idx])
    part_3 = f3[q_idx].bitwise_and(f3[kv_idx])
    part_4 = f4[q_idx].bitwise_and(f4[kv_idx])
    part_5 = f5[q_idx].bitwise_and(f5[kv_idx])
    sum = part_0 + part_1 + part_2 + part_3 + part_4 + part_5
    return torch.where(sum > 0, score, -float("inf"))


ref = scaled_dot_product_attention_with_flags(
    q,
    k,
    v,
    f,
)
kernel_options = {
    "BLOCK_M": 16,
    "BLOCK_N": 16,
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}
out = flex_attention(
    q,
    k,
    v,
    score_mod=dense_mod,
    kernel_options=kernel_options,
)

diff = (ref - out).abs()
diffmean = diff.mean()
diffmax = diff.max()
idx = diff.argmax()

ic(ref.shape)
ic(out.shape)
ic(idx)
ic(diffmean)
ic(diffmax)

ic(diff.flatten()[idx - 10 : idx + 10])
