cmake_minimum_required(VERSION 3.12)
project(KernelImageProcessing)

set(CMAKE_CXX_STANDARD 11)

add_executable(KernelImageProcessing main.cpp Image.cpp Image.h Pixel.h Kernel.cpp Kernel.h KernelFactory.cpp KernelFactory.h Filters/GaussianBlur.cpp Filters/GaussianBlur.h Filters/Identity.cpp Filters/Identity.h Filters/Sharpen.cpp Filters/Sharpen.h Filters/EdgeDetection.cpp Filters/EdgeDetection.h Filters/BoxBlur.cpp Filters/BoxBlur.h)