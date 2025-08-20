// bitwise_and_u64.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

__global__ void bitwise_and_u64_kernel(const uint64_t* __restrict__ a,
                                       const uint64_t* __restrict__ b,
                                       uint64_t* __restrict__ out,
                                       size_t n)
{
    const size_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride  = (size_t)blockDim.x * gridDim.x;

    // Fast path: 128-bit vectorized loads/stores when 16B aligned and even length.
    if ( ((reinterpret_cast<uintptr_t>(a) |
           reinterpret_cast<uintptr_t>(b) |
           reinterpret_cast<uintptr_t>(out)) & 0xF) == 0 &&
         (n & 1ULL) == 0 )
    {
        const size_t n2 = n >> 1; // number of ulonglong2s
        const ulonglong2* __restrict__ A = reinterpret_cast<const ulonglong2*>(a);
        const ulonglong2* __restrict__ B = reinterpret_cast<const ulonglong2*>(b);
        ulonglong2* __restrict__ O       = reinterpret_cast<ulonglong2*>(out);

        #pragma unroll 4
        for (size_t i = tid; i < n2; i += stride) {
            ulonglong2 va = A[i];
            ulonglong2 vb = B[i];
            O[i] = make_ulonglong2(va.x & vb.x, va.y & vb.y);
        }
    }
    else
    {
        // General path: grid-stride loop over 64-bit elements.
        #pragma unroll 4
        for (size_t i = tid; i < n; i += stride) {
            out[i] = a[i] & b[i];
        }
    }
}

void bitwise_and_u64(cudaStream_t stream,
                     const uint64_t* d_a,
                     const uint64_t* d_b,
                     uint64_t* d_out,
                     size_t n)
{
    if (n == 0) return;

    // Choose launch config via occupancy, then cap grid by available SMs.
    int blockSize = 0, minGridSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       bitwise_and_u64_kernel, 0, 0);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    // Aim for high occupancy without oversubscribing: e.g., 32 waves per SM.
    int blocksPerSM = (minGridSize + smCount - 1) / max(smCount, 1);
    if (blocksPerSM < 1) blocksPerSM = 1;
    if (blocksPerSM > 32) blocksPerSM = 32;

    int grid = smCount * blocksPerSM;

    // Lightweight cap to avoid launching far more threads than work items.
    const size_t maxUsefulBlocks =
        (n + (size_t)blockSize - 1) / (size_t)blockSize;
    if ((size_t)grid > maxUsefulBlocks) grid = (int)maxUsefulBlocks;
    if (grid < 1) grid = 1;

    bitwise_and_u64_kernel<<<grid, blockSize, 0, stream>>>(d_a, d_b, d_out, n);
}
