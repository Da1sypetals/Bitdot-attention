import torch
import packbits
from icecream import ic

a = torch.zeros((1, 67), dtype=torch.bool).cuda()
a[0, -1] = 1
a[0, 7] = 1

out1 = packbits.pack_bits(a, 2)
a[0, 3] = 1
out2 = packbits.pack_bits(a, 2)
ic(out1)
ic(out2)
ic(packbits.and_u64(out1, out2))

# test logic:
# pack -> and == and -> pack

a = torch.randint(0, 2, (300, 124), dtype=torch.bool).cuda()
b = torch.randint(0, 2, (300, 124), dtype=torch.bool).cuda()

q = packbits.pack_bits(a.logical_and(b), 2)
w = packbits.and_u64(
    packbits.pack_bits(a, 2),
    packbits.pack_bits(b, 2),
)

diff = q.to(torch.int32) - w.to(torch.int32)
ic(diff.abs().sum())
