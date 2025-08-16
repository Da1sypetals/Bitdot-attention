#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void pack_bits_kernel(
    const int32_t* __restrict__ _0,
    int32_t* __restrict__ _1,
    int _2, int _3, int _4, int _5
) {
    int _6  = (blockIdx.x << 8) | (threadIdx.x & 0xFF);
    int _7  = _6;
    if (_7 >= (_2 * _5)) return;

    int _8  = _7 / _5;
    int _9  = _7 % _5;

    int _10 = _9 * _4;
    int _11 = _10 + _4;
    if (_11 > _3) _11 = _3;

    int32_t _12 = 0;
    for (int _13 = _10; _13 < _11; ++_13) {
        int32_t _14 = _0[_8 * _3 + _13];
        _12 |= (_14 != 0) << (_13 - _10);
    }
    _1[_7] = _12;
}

void pack_bits_cuda_launcher(
    const int32_t* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    cudaStream_t stream = 0
) {
    const int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    const int total_chunks = n * n_chunks;

    const int threads_per_block = 256;
    const int total_threads = total_chunks;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    pack_bits_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        f_binary, out, n, d_f, max_chunk_bits, n_chunks
    );
}
