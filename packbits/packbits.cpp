#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int max_bits_per_chunk = 64;

void pack_bits_cuda_launcher(
    const bool* f_binary,  // device ptr, shape (n, d_f)
    uint64_t* out,             // device ptr, shape (n * n_chunks)
    int n, int d_f,
    int pack_dim, cudaStream_t stream
);

void bitwise_and_u64(cudaStream_t stream,
                     const uint64_t* d_a,
                     const uint64_t* d_b,
                     uint64_t* d_out,
                     size_t n);


torch::Tensor pack_bits_cuda(torch::Tensor f_binary, int pack_dim) {
    // You must make sure f_binary is either 0 or 1
    // Or you will get wrong results
    TORCH_CHECK(f_binary.is_cuda(), "f_binary must be a CUDA tensor, but got device: ", f_binary.device());
    TORCH_CHECK(f_binary.dtype() == torch::kBool, "f_binary must be bool, but got dtype: ", f_binary.dtype());

    TORCH_CHECK(f_binary.dim() == 2, "f_binary must have shape (n, d_f), but got shape: ", f_binary.sizes());
    int n   = f_binary.size(0);
    int d_f = f_binary.size(1);

    TORCH_CHECK(pack_dim * max_bits_per_chunk >= d_f, "pack_dim * max_bits_per_chunk must be >= d_f, but got pack_dim: ", pack_dim, ", max_bits_per_chunk: ", max_bits_per_chunk, ", d_f: ", d_f);

    auto opts = torch::TensorOptions().dtype(torch::kUInt64).device(f_binary.device());
    auto out = torch::empty({n, pack_dim}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    pack_bits_cuda_launcher(
        f_binary.data_ptr<bool>(),
        out.data_ptr<uint64_t>(),
        n, d_f, pack_dim, stream
    );

    return out;
}


torch::Tensor and_u64(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor, but got device: ", a.device());
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor, but got device: ", b.device());
    TORCH_CHECK(a.dtype() == torch::kUInt64, "a must be uint64, but got dtype: ", a.dtype());
    TORCH_CHECK(b.dtype() == torch::kUInt64, "b must be uint64, but got dtype: ", b.dtype());
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape, but got a shape: ", a.sizes(), ", b shape: ", b.sizes());

    auto out = torch::empty_like(a);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bitwise_and_u64(
        stream,
        a.data_ptr<uint64_t>(),
        b.data_ptr<uint64_t>(),
        out.data_ptr<uint64_t>(),
        a.numel()
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_bits", &pack_bits_cuda, "Pack Bits: bool (n, d_f) -> uint64 (n, pack_dim)");
    m.def("and_u64", &and_u64, "Bitwise and u64");
}