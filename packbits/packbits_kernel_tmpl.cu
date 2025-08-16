#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Template kernel: each thread handles WorkPerThread chunks
template<int WorkPerThread>
__global__ void pack_bits_kernel(
    const int32_t* __restrict__ f_binary,  // (n, d_f)
    int32_t* __restrict__ out,             // (n * n_chunks)
    int n, int d_f, int max_chunk_bits, int n_chunks
) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int start_chunk_idx = global_thread_id * WorkPerThread;

#pragma unroll
    for (int w = 0; w < WorkPerThread; ++w) {
        int flat_chunk_idx = start_chunk_idx + w;  // 0 .. n*n_chunks-1
        if (flat_chunk_idx >= n * n_chunks) return;

        int row      = flat_chunk_idx / n_chunks;
        int chunk_id = flat_chunk_idx % n_chunks;

        int start = chunk_id * max_chunk_bits;
        int end   = min(start + max_chunk_bits, d_f);

        int32_t packed_val = 0;
        for (int bit = start; bit < end; ++bit) {
            int32_t bit_val = f_binary[row * d_f + bit];
            if (bit_val != 0) {
                packed_val |= (1 << (bit - start));
            }
        }
        out[flat_chunk_idx] = packed_val;
    }
}

void pack_bits_cuda_launcher(
    const int32_t* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    cudaStream_t stream = 0
) {
    constexpr int WorkPerThread = 1;
    int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    int total_chunks = n * n_chunks;

    int threads_per_block = 256;
    int total_threads = (total_chunks + WorkPerThread - 1) / WorkPerThread;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    pack_bits_kernel<WorkPerThread><<<num_blocks, threads_per_block, 0, stream>>>(
        f_binary, out, n, d_f, max_chunk_bits, n_chunks
    );
}
