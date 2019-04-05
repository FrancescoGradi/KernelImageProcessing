#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <curand_mtgp32_kernel.h>

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define TILE_WIDTH 32
#define w (TILE_WIDTH + 3 - 1)

__global__ void naiveFiltering(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                               int n, int widthResult, int heightResult, int channels) {

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    float sum;
    int a, b;

    for(int i = 0; i < 3; i++) {

        if ((row < heightResult) && (col < widthResult)) {

            sum = 0;
            a = 0;

            for (int k = row; k < row + n; k++) {
                b = 0;

                for (int l = col; l < col + n; l++) {
                    sum += kernelDevice[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
                    b++;
                }
                a++;
            }

            if (sum < 0)
                sum = 0;
            if (sum > 1)
                sum = 1;

            resultDevice[row * widthResult * channels + col * channels + i] = sum;

        }
    }
}

__global__ void tiling(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                       int n, int widthResult, int heightResult, int channels) {
    __shared__ float N_ds[w][w];

    for (int k = 0; k < channels; ++k) {

        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - (n/2);
        int srcX = blockIdx.x * TILE_WIDTH + destX - (n/2);
        int src = srcY*width*channels + srcX*channels + k;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = pixelsDevice[src];
        } else {
            N_ds[destY][destX] = 0;
        }

        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - (n/2);
        srcX = blockIdx.x * TILE_WIDTH + destX - (n/2);
        src = srcY*width*channels + srcX*channels + k;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                N_ds[destY][destX] = pixelsDevice[src];
            } else {
                N_ds[destY][destX] = 0;
            }
        }
        __syncthreads();

        float sum = 0;
        int y, x;
        for (y = 0; y < n; ++y) {
            for (x = 0; x < n; ++x) {
                sum += N_ds[threadIdx.y + y][threadIdx.x + x] * kernelDevice[y * n + x];
            }
        }

        if (sum < 0)
            sum = 0;
        if (sum > 1)
            sum = 1;

        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;

        if (y < heightResult && x < widthResult)
            resultDevice[y*widthResult*channels + x*channels + k] = sum;
        __syncthreads();
    }
}

__global__ void tilingConstant(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
                       int n, int widthResult, int heightResult, int channels) {

    float N_ds[w][w];

    for (int k = 0; k < channels; ++k) {

        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - (n/2);
        int srcX = blockIdx.x * TILE_WIDTH + destX - (n/2);
        int src = srcY*width*channels + srcX*channels + k;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = pixelsDevice[src];
        } else {
            N_ds[destY][destX] = 0;
        }

        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - (n/2);
        srcX = blockIdx.x * TILE_WIDTH + destX - (n/2);
        src = srcY*width*channels + srcX*channels + k;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                N_ds[destY][destX] = pixelsDevice[src];
            } else {
                N_ds[destY][destX] = 0;
            }
        }
        __syncthreads();

        float sum = 0;
        int y, x;
        for (y = 0; y < n; ++y) {
            for (x = 0; x < n; ++x) {
                sum += N_ds[threadIdx.y + y][threadIdx.x + x] * kernelDevice[y * n + x];
            }
        }

        if (sum < 0)
            sum = 0;
        if (sum > 1)
            sum = 1;

        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;

        if (y < heightResult && x < widthResult)
            resultDevice[y*widthResult*channels + x*channels + k] = sum;
        __syncthreads();
    }
}