#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <stdexcept>

__global__ void pack_bits_float_kernel(
    const float* __restrict__ fb,
    int32_t* __restrict__ o,
    int n, int d, int cb, int nc
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n * nc) return;
    int b = a / nc, e = a % nc * cb;
    int f = min(e + cb, d);
    int32_t g = 0;
    const uint32_t* u = reinterpret_cast<const uint32_t*>(fb);
    for (int h = e; h < f; ++h) g |= (static_cast<bool>(u[b * d + h] & 0x7FFFFFFF) << (h - e));
    o[a] = g;
}

void pack_bits_float_cuda_launcher(
    const float* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    cudaStream_t stream = 0
) {
    const int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    const int total_chunks = n * n_chunks;

    const int threads_per_block = 256;
    const int total_threads = total_chunks;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    pack_bits_float_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        f_binary, out, n, d_f, max_chunk_bits, n_chunks
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

}

// __global__ void pack_bits_float_kernel(
//     const float* __restrict__ fb,   // (n, d)
//     int32_t* __restrict__ o,        // (n * nc)
//     int n, int d, int cb, int nc
// ) {
//     int a = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. n*nc-1
//     if (a >= n * nc) return;

//     int b = a / nc;  // row index
//     int c = a % nc;  // chunk index within row

//     int e = c * cb;
//     int f = min(e + cb, d);

//     const float* row_ptr = fb + b * d + e;
//     int len = f - e;

//     int32_t g = 0;
//     int i = 0;

//     // --- peel until 16-byte aligned ---
//     uintptr_t addr = reinterpret_cast<uintptr_t>(row_ptr);
//     while (i < len && (addr & 0xF)) {
//         int32_t x = __float2int_rz(row_ptr[i]);
//         g |= (x << i);
//         i++;
//         addr += sizeof(float);
//     }

//     // --- vectorized body (float4 loads) ---
//     for (; i + 4 <= len; i += 4) {
//         float4 v = *reinterpret_cast<const float4*>(row_ptr + i);
//         int32_t x0 = __float2int_rz(v.x);
//         int32_t x1 = __float2int_rz(v.y);
//         int32_t x2 = __float2int_rz(v.z);
//         int32_t x3 = __float2int_rz(v.w);

//         g |= (x0 << (i + 0));
//         g |= (x1 << (i + 1));
//         g |= (x2 << (i + 2));
//         g |= (x3 << (i + 3));
//     }

//     // --- tail ---
//     for (; i < len; i++) {
//         int32_t x = __float2int_rz(row_ptr[i]);
//         g |= (x << i);
//     }

//     o[a] = g;
// }
