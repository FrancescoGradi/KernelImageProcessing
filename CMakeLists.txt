cmake_minimum_required(VERSION 3.12)
project(KernelImageProcessing)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(KernelImageProcessing Image.cpp Image.h Kernel.cpp Kernel.h KernelFactory.cpp
        KernelFactory.h Filters/Identity.cpp Filters/Identity.h Filters/GaussianBlur.cpp Filters/GaussianBlur.h
        Filters/Sharpen.cpp Filters/Sharpen.h Filters/EdgeDetection.cpp Filters/EdgeDetection.h Filters/BoxBlur.cpp
        Filters/BoxBlur.h main.cu speedTests.cu CUDAError.h filteringUtils.cu filteringUtils.h speedTests.h)

