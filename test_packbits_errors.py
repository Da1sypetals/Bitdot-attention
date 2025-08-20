import torch
import numpy as np
from packbits import pack_bits, and_u64


# Real-world binary data (e.g., image masks or feature matrices)
def generate_real_binary_data(shape):
    return torch.from_numpy(np.random.randint(0, 2, size=shape, dtype=bool)).cuda()


# Real-world uint64 data (e.g., hash values or bit masks)
def generate_real_uint64_data(shape):
    return torch.from_numpy(np.random.randint(0, 2**64, size=shape, dtype=np.uint64)).cuda()


def test_pack_bits_errors():
    # Test non-CUDA tensor
    try:
        f_binary = generate_real_binary_data((100, 64)).cpu()
        pack_bits(f_binary, 1)
    except Exception as e:
        print(f"Test non-CUDA tensor: {e}")

    # Test non-bool tensor
    try:
        f_binary = generate_real_uint64_data((100, 64)).to(torch.float32)
        pack_bits(f_binary, 1)
    except Exception as e:
        print(f"Test non-bool tensor: {e}")

    # Test non-2D tensor
    try:
        f_binary = generate_real_binary_data((100, 64, 3))
        pack_bits(f_binary, 1)
    except Exception as e:
        print(f"Test non-2D tensor: {e}")

    # Test invalid pack_dim
    try:
        f_binary = generate_real_binary_data((100, 266))
        pack_bits(f_binary, 4)  # pack_dim=1 is too small for d_f=128
    except Exception as e:
        print(f"Test invalid pack_dim: {e}")


def test_and_u64_errors():
    # Test non-CUDA tensor for a
    try:
        a = generate_real_uint64_data((100,)).cpu()
        b = generate_real_uint64_data((100,))
        and_u64(a, b)
    except Exception as e:
        print(f"Test non-CUDA tensor for a: {e}")

    # Test non-CUDA tensor for b
    try:
        a = generate_real_uint64_data((100,))
        b = generate_real_uint64_data((100,)).cpu()
        and_u64(a, b)
    except Exception as e:
        print(f"Test non-CUDA tensor for b: {e}")

    # Test non-uint64 tensor for a
    try:
        a = generate_real_binary_data((100,)).to(torch.float32)
        b = generate_real_uint64_data((100,))
        and_u64(a, b)
    except Exception as e:
        print(f"Test non-uint64 tensor for a: {e}")

    # Test non-uint64 tensor for b
    try:
        a = generate_real_uint64_data((100,))
        b = generate_real_binary_data((100,)).to(torch.float32)
        and_u64(a, b)
    except Exception as e:
        print(f"Test non-uint64 tensor for b: {e}")

    # Test shape mismatch
    try:
        a = generate_real_uint64_data((100,))
        b = generate_real_uint64_data((50,))
        and_u64(a, b)
    except Exception as e:
        print(f"Test shape mismatch: {e}")


if __name__ == "__main__":
    test_pack_bits_errors()
    test_and_u64_errors()
