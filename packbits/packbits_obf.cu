#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void pack_bits_kernel(
    const bool* __restrict__ f_binary,  // (n, d_f)
    int32_t* __restrict__ out,             // (n * n_chunks)
    int n, int d_f, int max_chunk_bits, int n_chunks
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n * n_chunks) return;
    int d = c / n_chunks;
    int e = c % n_chunks;
    int f = e * max_chunk_bits;
    int g = min(f + max_chunk_bits, d_f);
    int32_t h = 0;
    for (int i = f; i < g; ++i) {
        int32_t j = (int32_t)(f_binary[d * d_f + i]);
        h |= (j << (i - f));
    }
    out[c] = h;
}

void pack_bits_cuda_launcher(
    const bool* f_binary,  // device ptr, shape (n, d_f)
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
