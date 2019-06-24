//
// Created by fra on 05/04/19.
//

#ifndef KERNELIMAGEPROCESSING_CONSTANTMEMORYUTILS_H
#define KERNELIMAGEPROCESSING_CONSTANTMEMORYUTILS_H

#include <host_defines.h>
#include <string>

__global__ void constantMemoryFiltering(float* pixelsDevice, float* resultDevice, int width, int height,
                                        int n, int widthResult, int heightResult, int channels);

__global__ void constantMemoryFilteringIdentity(float* pixelsDevice, float* resultDevice, int width, int height,
                                        int n, int widthResult, int heightResult, int channels);
__global__ void constantMemoryFilteringBlur(float* pixelsDevice, float* resultDevice, int width, int height,
                                                int n, int widthResult, int heightResult, int channels);
__global__ void constantMemoryFilteringBoxBlur(float* pixelsDevice, float* resultDevice, int width, int height,
                                                int n, int widthResult, int heightResult, int channels);
__global__ void constantMemoryFilteringEdge(float* pixelsDevice, float* resultDevice, int width, int height,
                                                int n, int widthResult, int heightResult, int channels);
__global__ void constantMemoryFilteringSharpen(float* pixelsDevice, float* resultDevice, int width, int height,
                                                int n, int widthResult, int heightResult, int channels);

double CUDAConstantMemory(int kernelSize, std::string imagePath, std::string filterName);
double CUDAConstantMemory(int kernelSize, std::string imagePath);

#endif //KERNELIMAGEPROCESSING_CONSTANTMEMORYUTILS_H
