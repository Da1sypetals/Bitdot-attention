from tqdm import trange
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import packbits

flex_attention = torch.compile(flex_attention)


def scaled_dot_product_attention_with_flags(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    flag: torch.Tensor,
) -> torch.Tensor:
    flag_dot_products = torch.matmul(flag, flag.transpose(-2, -1))
    flag_mask = flag_dot_products > 0.0

    output = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=flag_mask,
        dropout_p=0.0,
        is_causal=False,
    )

    return output


def gen_input(b, h, n, d, d_f, max_chunk_bits):
    device = "cuda"

    q = torch.randn(b, h, n, d, device=device)
    k = torch.randn(b, h, n, d, device=device)
    v = torch.randn(b, h, n, d, device=device)

    f = torch.zeros(n * d_f, device=device, dtype=torch.float32)
    perm = torch.randperm(n * d_f, device=device)[: int(n * d_f * 0.05)]
    f[perm] = 1.0
    f = f.reshape(1, 1, n, d_f)

    f_binary = (f > 0).squeeze(0).squeeze(0).to(torch.bool)  # (n, d_f)

    fint = packbits.pack_bits(f_binary, max_chunk_bits)

    return q, k, v, f, fint


b = 1
h = 8
n = 5013
d = 192
d_f = 151
max_chunk_bits = 28
print(f"Using {max_chunk_bits = }")

q, k, v, f, fint = gen_input(
    b,
    h,
    n,
    d,
    d_f,
    max_chunk_bits,
)


def dense_mod(score, b, h, q_idx, kv_idx):
    part_0 = fint[q_idx * 6 + 0].bitwise_and(fint[kv_idx * 6 + 0])
    part_1 = fint[q_idx * 6 + 1].bitwise_and(fint[kv_idx * 6 + 1])
    part_2 = fint[q_idx * 6 + 2].bitwise_and(fint[kv_idx * 6 + 2])
    part_3 = fint[q_idx * 6 + 3].bitwise_and(fint[kv_idx * 6 + 3])
    part_4 = fint[q_idx * 6 + 4].bitwise_and(fint[kv_idx * 6 + 4])
    part_5 = fint[q_idx * 6 + 5].bitwise_and(fint[kv_idx * 6 + 5])
    sum = part_0 + part_1 + part_2 + part_3 + part_4 + part_5
    return torch.where(sum > 0, score, -float("inf"))


# warmup
for _ in range(100):
    _ = q.matmul(k.transpose(-2, -1))
torch.cuda.synchronize()

N_REP = 20
diffmean_list = []
diffmax_list = []

# baseline reference
ref = scaled_dot_product_attention_with_flags(q, k, v, f)

kernel_options = {
    "BLOCK_M": 16,
    "BLOCK_N": 16,
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}

# warmup flex (compile)
for _ in trange(3):
    out = flex_attention(
        q,
        k,
        v,
        score_mod=dense_mod,
        kernel_options=kernel_options,
    )

torch.cuda.synchronize()

# main loop
for _ in trange(N_REP):
    q, k, v, f, fint = gen_input(
        b,
        h,
        n,
        d,
        d_f,
        max_chunk_bits,
    )
    ref = scaled_dot_product_attention_with_flags(q, k, v, f)
    out = flex_attention(
        q,
        k,
        v,
        score_mod=dense_mod,
        kernel_options=kernel_options,
    )

    diff = (ref - out).abs()
    diffmean = diff.mean().item()
    diffmax = diff.max().item()

    diffmean_list.append(diffmean)
    diffmax_list.append(diffmax)

for i, (diffmean, diffmax) in enumerate(zip(diffmean_list, diffmax_list)):
    # make width consistent
    print(f"Exp {i:03d} | diffmean={diffmean:.3e} | diffmax={diffmax:.3e}")
