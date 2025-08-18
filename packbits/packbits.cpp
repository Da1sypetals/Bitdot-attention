#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>


void pack_bits_cuda_launcher(
    const bool* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    cudaStream_t stream
);

void pack_bits_float_cuda_launcher(
    const float* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    int pack_dim, cudaStream_t stream
);

torch::Tensor pack_bits_cuda(torch::Tensor f_binary, int max_chunk_bits) {
    // You must make sure f_binary is either 0 or 1
    // Or you will get wrong results
    TORCH_CHECK(f_binary.is_cuda(), "f_binary must be a CUDA tensor");
    TORCH_CHECK(f_binary.dtype() == torch::kBool, "f_binary must be bool");

    TORCH_CHECK(f_binary.dim() == 2, "f_binary must have shape (n, d_f)");
    int n   = f_binary.size(0);
    int d_f = f_binary.size(1);

    int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(f_binary.device());
    auto out = torch::empty({n * n_chunks}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    pack_bits_cuda_launcher(
        f_binary.data_ptr<bool>(),
        out.data_ptr<int32_t>(),
        n, d_f, max_chunk_bits,
        stream
    );

    return out;
}

// input: (n, d_f) float32
// output: (n, pack_dim) int32
torch::Tensor pack_bits_float_cuda(torch::Tensor f_binary, int max_chunk_bits, int pack_dim) {
    // You must make sure f_binary is either 0 or 1
    // Or you will get wrong results
    TORCH_CHECK(f_binary.is_cuda(), "f_binary must be a CUDA tensor");
    TORCH_CHECK(f_binary.dtype() == torch::kFloat, "f_binary must be float32");

    TORCH_CHECK(f_binary.dim() == 2, "f_binary must have shape (n, d_f)");
    int n   = f_binary.size(0);
    int d_f = f_binary.size(1);

    int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    TORCH_CHECK(n_chunks <= pack_dim, "# of chunk < pack_dim required");
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(f_binary.device());
    // Unfilled positions are automatically padded with zeros
    auto out = torch::zeros({n * pack_dim}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    pack_bits_float_cuda_launcher(
        f_binary.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        n, d_f, max_chunk_bits,
        pack_dim, stream
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_bits", &pack_bits_cuda, "Pack Bits (n, d_f) -> (n * n_chunks)");
    m.def("pack_bits_float", &pack_bits_float_cuda, "Pack Bits Float32 (n, d_f) -> (n * n_chunks)");
}