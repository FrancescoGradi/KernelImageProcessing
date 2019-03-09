cmake_minimum_required(VERSION 3.12)
project(KernelImageProcessing)

set(CMAKE_CXX_STANDARD 11)

add_executable(KernelImageProcessing main.cpp Image.cpp Image.h Pixel.h Kernel.cpp Kernel.h)