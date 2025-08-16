#include <torch/extension.h>

// Declare the CUDA implementation (from attention_kernel.cu)
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FlashAttention Forward (CUDA)");
}
