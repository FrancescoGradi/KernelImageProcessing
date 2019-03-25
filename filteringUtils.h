//
// Created by federico on 25/03/19.
//

#ifndef KERNELIMAGEPROCESSING_FILTERINGUTILS_H
#define KERNELIMAGEPROCESSING_FILTERINGUTILS_H

__global__ void naiveFiltering(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                               int n, int widthResult, int heightResult, int channels);
__global__ void tilingFiltering(int* intPixelsRed, int* intPixelsGreen, int* intPixelsBlue, float* kernelDevice,
                                Pixel* resultDevice, int width, int height, int n, int widthResult, int heightResult);
__global__ void tiling(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                       int n, int widthResult, int heightResult, int channels);

#endif //KERNELIMAGEPROCESSING_FILTERINGUTILS_H
