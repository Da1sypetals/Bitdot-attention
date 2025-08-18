#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <stdexcept>

__global__ void pack_bits_float_kernel(
    const float* __restrict__ fb,
    int32_t* __restrict__ o,
    int n, int d, int cb, int nc, int pd
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n * nc) return;
    int b = a / nc, c = a % nc;
    int e = c * cb;
    int f = min(e + cb, d);
    int32_t g = 0;
    const uint32_t* u = reinterpret_cast<const uint32_t*>(fb);
    for (int h = e; h < f; ++h) g |= (static_cast<bool>(u[b * d + h] & 0x7FFFFFFF) << (h - e));
    o[b * pd + c] = g;
}

void pack_bits_float_cuda_launcher(
    const float* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    int pack_dim, cudaStream_t stream = 0
) {
    const int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    const int total_chunks = n * n_chunks;

    const int threads_per_block = 256;
    const int total_threads = total_chunks;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    pack_bits_float_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        f_binary, out, n, d_f, max_chunk_bits, n_chunks, pack_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

}
