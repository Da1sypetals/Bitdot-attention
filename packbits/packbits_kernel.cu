#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void pack_bits_kernel(
    const bool* __restrict__ f_binary,  // (n, d_f)
    int32_t* __restrict__ out,             // (n * n_chunks)
    int n, int d_f, int max_chunk_bits, int n_chunks
) {
    int flat_chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. n*n_chunks-1
    if (flat_chunk_idx >= n * n_chunks) return;

    int row      = flat_chunk_idx / n_chunks;
    int chunk_id = flat_chunk_idx % n_chunks;

    int start = chunk_id * max_chunk_bits;
    int end   = min(start + max_chunk_bits, d_f);

    int32_t packed_val = 0;
    for (int bit = start; bit < end; ++bit) {
        int32_t bit_val = (int32_t)(f_binary[row * d_f + bit]);
        packed_val |= (bit_val << (bit - start));
    }
    out[flat_chunk_idx] = packed_val;
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

