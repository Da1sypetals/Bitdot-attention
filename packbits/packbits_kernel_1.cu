

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef LDG
// Use __ldg if available (read-only cache). Otherwise fall back to plain load.
#if __CUDACC_VER_MAJOR__ >= 7
#define LDG(x) (__ldg(x))
#else
#define LDG(x) (*(x))
#endif
#endif

template<int WorkPerThread>
__global__ void pack_bits_kernel_optimized(
    const int32_t* __restrict__ f_binary,  // (n, d_f) - values 0/1 in low bit
    int32_t* __restrict__ out,             // (n * n_chunks)
    int n, int d_f, int max_chunk_bits, int n_chunks,
    int total_chunks
) {
    // global thread id
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int start_chunk_idx = global_thread_id * WorkPerThread;

    // Each thread handles up to WorkPerThread chunks
    for (int w = 0; w < WorkPerThread; ++w) {
        int flat_chunk_idx = start_chunk_idx + w;
        if (flat_chunk_idx >= total_chunks) break; // safer than return inside loop

        // compute row and chunk_id
        int row      = flat_chunk_idx / n_chunks;
        int chunk_id = flat_chunk_idx - row * n_chunks; // faster than % on some archs

        int start = chunk_id * max_chunk_bits;
        int end   = start + max_chunk_bits;
        if (end > d_f) end = d_f;

        const int32_t* row_ptr = f_binary + row * d_f;

        unsigned int packed_val = 0u;

        int bits = end - start;
        int i = 0;

        // Unroll 4 bits at a time
        for (; i + 3 < bits; i += 4) {
            // load 4 int32 values; use LDG for read-only cache (macro falls back if absent)
            int32_t v0 = (int32_t)LDG(row_ptr + start + i + 0);
            int32_t v1 = (int32_t)LDG(row_ptr + start + i + 1);
            int32_t v2 = (int32_t)LDG(row_ptr + start + i + 2);
            int32_t v3 = (int32_t)LDG(row_ptr + start + i + 3);

            // mask LSB and shift into place
            packed_val |= ((unsigned int)(v0 & 1) << (i + 0));
            packed_val |= ((unsigned int)(v1 & 1) << (i + 1));
            packed_val |= ((unsigned int)(v2 & 1) << (i + 2));
            packed_val |= ((unsigned int)(v3 & 1) << (i + 3));
        }

        // tail
        for (; i < bits; ++i) {
            int32_t v = (int32_t)LDG(row_ptr + start + i);
            packed_val |= ((unsigned int)(v & 1) << i);
        }

        out[flat_chunk_idx] = (int32_t)packed_val;
    }
}

void pack_bits_cuda_launcher(
    const int32_t* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    cudaStream_t stream = 0
) {
    constexpr int WorkPerThread = 1; // same default you used; tune if needed
    int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    int total_chunks = n * n_chunks;

    int threads_per_block = 256; // try 128 or 256 depending on GPU; profile to pick best
    int total_threads = (total_chunks + WorkPerThread - 1) / WorkPerThread;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Launch optimized kernel
    pack_bits_kernel_optimized<WorkPerThread><<<num_blocks, threads_per_block, 0, stream>>>(
        f_binary, out, n, d_f, max_chunk_bits, n_chunks, total_chunks
    );
}