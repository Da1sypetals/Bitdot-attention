from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="attention_cuda",
            sources=[
                "attention.cpp",  # wrapper
                "flash.cu",  # kernel
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
