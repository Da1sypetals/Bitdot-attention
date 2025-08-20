#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <stdexcept>

constexpr int max_bits_per_chunk = 64;

__global__ void pack_bits_float_kernel(
    const bool* __restrict__ f_binary,
    uint64_t* __restrict__ out,
    int num_rows, int d_f, int num_chunk, int pack_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows * num_chunk) return;
    int row_idx = tid / num_chunk, chunk_idx = tid % num_chunk;
    int start = chunk_idx * max_bits_per_chunk;
    int end = min(start + max_bits_per_chunk, d_f);
    uint64_t res = 0;
    for (int bit = start; bit < end; ++bit) {
        uint64_t bit_val = static_cast<uint64_t>(f_binary[row_idx * max_bits_per_chunk + bit]);
        res |= (bit_val << (bit - start));
    }
    out[row_idx * num_chunk + chunk_idx] = res;
}

void pack_bits_cuda_launcher(
    const bool* f_binary,  // device ptr, shape (n, d_f)
    uint64_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f,
    int pack_dim, cudaStream_t stream
) {
    const int n_chunks = (d_f + max_bits_per_chunk - 1) / max_bits_per_chunk;
    const int total_chunks = n * n_chunks;

    constexpr int threads_per_block = 256;
    const int total_threads = total_chunks;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    pack_bits_float_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        f_binary, out, n, d_f, n_chunks, pack_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

}
