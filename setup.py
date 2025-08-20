from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="packbits",
    ext_modules=[
        CUDAExtension(
            name="packbits",
            sources=[
                "packbits/packbits.cpp",  # wrapper
                "packbits/packbits_kernel.cu",  # kernel
                "packbits/and.cu",  # kernel
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
