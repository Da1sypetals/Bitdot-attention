from icecream import ic
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# flex_attention = torch.compile(flex_attention)


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


ref = scaled_dot_product_attention_with_flags(
    q,
    k,
    v,
    f,
)

f = f[0, 0]


def dense_mod(score, b, h, q_idx, kv_idx):
    return torch.where(
        torch.tensor(f[q_idx].dot(f[kv_idx])) > 0.0,
        score,
        -float("inf"),
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

diff = (ref - out).abs().mean()
idx = (ref - out).abs().argmax()

ic(ref.shape)
ic(out.shape)
ic(idx)
print(f"diff = {diff.item(): .9f}")
ic(ref[0, 0, 0, :10])
ic(out[0, 0, 0, :10])

# ic(ref[0, 1, 1, :10])
# ic(out[0, 1, 1, :10])

# ic(ref.flatten()[idx - 10 : idx + 10])
# ic(out.flatten()[idx - 10 : idx + 10])
