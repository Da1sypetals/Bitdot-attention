#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>


void pack_bits_cuda_launcher(
    const int32_t* f_binary,  // device ptr, shape (n, d_f)
    int32_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f, int max_chunk_bits,
    cudaStream_t stream
);

torch::Tensor pack_bits_cuda(torch::Tensor f_binary, int max_chunk_bits) {
    // You must make sure f_binary is either 0 or 1
    // Or you will get wrong results
    TORCH_CHECK(f_binary.is_cuda(), "f_binary must be a CUDA tensor");
    TORCH_CHECK(f_binary.dtype() == torch::kInt32, "f_binary must be int32");

    TORCH_CHECK(f_binary.dim() == 2, "f_binary must have shape (n, d_f)");
    int n   = f_binary.size(0);
    int d_f = f_binary.size(1);

    int n_chunks = (d_f + max_chunk_bits - 1) / max_chunk_bits;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(f_binary.device());
    auto out = torch::empty({n * n_chunks}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    pack_bits_cuda_launcher(
        f_binary.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(),
        n, d_f, max_chunk_bits,
        stream
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_bits", &pack_bits_cuda, "Pack Bits (n, d_f) -> (n * n_chunks)");
}