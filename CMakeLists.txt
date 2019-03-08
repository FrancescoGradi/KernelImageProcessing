cmake_minimum_required(VERSION 3.13)
project(KernelImageProcessing)

set(CMAKE_CXX_STANDARD 11)

add_executable(KernelImageProcessing main.cpp Image.cpp Image.h Image.cpp Image.h Pixel.h)