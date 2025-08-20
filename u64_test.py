import torch
import packbits
from icecream import ic

a = torch.zeros((1, 65), dtype=torch.float32).cuda()
a[0, -1] = 1
a[0, 33] = 1

out1 = packbits.pack_bits_float_u64(a, 2)
a[0, 0] = 1
out2 = packbits.pack_bits_float_u64(a, 2)
ic(out1)
ic(out2)
ic(packbits.and_u64(out1, out2))
