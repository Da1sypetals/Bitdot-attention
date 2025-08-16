#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ===== Compile-time constants =====
constexpr int BLOCK = 8;
constexpr int SRAM_QKV_CHUNKS = 3;
constexpr float NEG_INF = -INFINITY;

__global__
void forward_kernel(const float* Q, const float* K, const float* V,
                    const int N, const int d,
                    const int Tc, const int Tr,
                    const float softmax_scale,
                    float* l, float* m, float* O) {

    const int tx = threadIdx.x;
    const int bx = blockIdx.x; 
    const int by = blockIdx.y; // batch and head index

    // Offsets for each batch and head
    const int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    const int lm_offset  = (bx * gridDim.y * N) + (by * N);

    // Shared memory layout: Qi, Kj, Vj, S
    extern __shared__ float sram[];
    const int tile_size = BLOCK * d;

    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * SRAM_QKV_CHUNKS];

    for (int j = 0; j < Tc; ++j) {

        // Load Kj, Vj to shared memory
        for (int x = 0; x < d; ++x) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; ++i)  {

            // Load Qi, l, m
            for (int x = 0; x < d; ++x) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (BLOCK * i) + tx];
            float row_l_prev = l[lm_offset + (BLOCK * i) + tx];

            // Compute S = QK^T
            float row_m = NEG_INF;
            for (int y = 0; y < BLOCK; ++y) {
                float sum = 0.0f;
                for (int x = 0; x < d; ++x) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(BLOCK * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // Softmax step: P = exp(S - row_m)
            float row_l = 0.0f;
            for (int y = 0; y < BLOCK; ++y) {
                S[(BLOCK * tx) + y] = __expf(S[(BLOCK * tx) + y] - row_m);
                row_l += S[(BLOCK * tx) + y];
            }

            // Update running max and sum
            const float row_m_new = fmaxf(row_m_prev, row_m);
            const float row_l_new =
                (__expf(row_m_prev - row_m_new) * row_l_prev) +
                (__expf(row_m - row_m_new) * row_l);

            // Write back O
            for (int x = 0; x < d; ++x) {
                float pv = 0.0f; // Pij * Vj
                for (int y = 0; y < BLOCK; ++y) {
                    pv += S[(BLOCK * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1.0f / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) *
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
                     (__expf(row_m - row_m_new) * pv));
            }

            m[lm_offset + (BLOCK * i) + tx] = row_m_new;
            l[lm_offset + (BLOCK * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int N  = Q.size(2);
    const int d  = Q.size(3);

    const int T = (N + BLOCK - 1) / BLOCK;
    constexpr float softmax_scale_factor = 1.0f; // we'll divide by sqrt(d) later

    const float softmax_scale = softmax_scale_factor / sqrtf(static_cast<float>(d));

    // Allocate output and intermediate buffers
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto m = torch::full({B, nh, N}, NEG_INF, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Shared memory size requirement
    const int sram_size =
        (SRAM_QKV_CHUNKS * BLOCK * d * sizeof(float)) +
        (BLOCK * BLOCK * sizeof(float));

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);   // batch_size x num_heads
    dim3 block_dim(BLOCK);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, T, T, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    return O;
}
