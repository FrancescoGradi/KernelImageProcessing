//
// Created by federico on 25/03/19.
//

#ifndef KERNELIMAGEPROCESSING_FILTERINGUTILS_H
#define KERNELIMAGEPROCESSING_FILTERINGUTILS_H

#include <host_defines.h>

__global__ void naiveFiltering(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                               int n, int widthResult, int heightResult, int channels);
__global__ void tiling(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                       int n, int widthResult, int heightResult, int channels);
__global__ void tilingConstant(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                       int n, int widthResult, int heightResult, int channels);

#endif //KERNELIMAGEPROCESSING_FILTERINGUTILS_H
