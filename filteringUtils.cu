#include "Pixel.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <curand_mtgp32_kernel.h>

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define TILE_WIDTH 8
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

__global__ void tilingFiltering(int* intPixelsRed, int* intPixelsGreen, int* intPixelsBlue, float* kernelDevice,
                                Pixel* resultDevice, int width, int height, int n, int widthResult, int heightResult) {
    /* Siccome viene utilizzata la shared memory, secondo me è più sensato fare le conversioni prima dell'allocazione
     * della memoria stessa. Purtroppo si perde un po' in velocità, perché dovranno essere fatti tre cicli al posto
     * di uno solo. Un'idea alternativa portebbe prevedere di scrivere una funzione che permetta di convertire
     * direttamente l'array di pixels in int/float, così da non dover effettuare la conversione nelle varie funzioni
     */
    // Se dichiari la variabile come extern, nella chiamata al kernel devi esplicitare la quantità di memoria da
    // allocare.
    __shared__ int sharedMemory[TILE_WIDTH];

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int a, b;
    int sum = 0;

    // Il processo ha bisogno di caricare un numero di chunk pari a (width * height) / TILE_WIDTH per ogni canale,
    // quindi avendo immagini a 3 canali devo ripetere il tutto per tre volte.
    for(int phase = 0; phase < (width * height) / TILE_WIDTH; ++phase) {
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sharedMemory[i] = intPixelsRed[i];
        }
        __syncthreads();

        if ((row < heightResult) && (col < widthResult)) {
            sum = 0;
            a = 0;

            for (int k = row; k < row + n; k++) {
                b = 0;

                for (int l = col; l < col + n; l++) {
                    sum += kernelDevice[a * n + b] * sharedMemory[k * width + l];
                    b++;
                }
                a++;
            }

            if (sum < 0)
                sum = 0;
            if (sum > 255)
                sum = 255;

            resultDevice[row * widthResult + col].r = ((char) sum);
        }
    }
    __syncthreads();
    for(int phase = 0; phase < (width * height) / TILE_WIDTH; ++phase) {
        // Ripeto il procedimento per il green
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sharedMemory[i] = intPixelsGreen[i];
        }
        __syncthreads();
        if ((row < heightResult) && (col < widthResult)) {
            sum = 0;
            a = 0;

            for (int k = row; k < row + n; k++) {
                b = 0;

                for (int l = col; l < col + n; l++) {
                    sum += kernelDevice[a * n + b] * sharedMemory[k * width + l];
                    b++;
                }
                a++;
            }

            if (sum < 0)
                sum = 0;
            if (sum > 255)
                sum = 255;

            resultDevice[row * widthResult + col].g = ((char) sum);
        }
    }
    __syncthreads();
    for(int phase = 0; phase < (width * height) / TILE_WIDTH; ++phase) {
        // Ripeto il procedimento per il blue
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sharedMemory[i] = intPixelsBlue[i];
        }
        __syncthreads();
        if ((row < heightResult) && (col < widthResult)) {
            sum = 0;
            a = 0;

            for (int k = row; k < row + n; k++) {
                b = 0;

                for (int l = col; l < col + n; l++) {
                    sum += kernelDevice[a*n + b] * sharedMemory[k*width + l];
                    b++;
                }
                a++;
            }

            if (sum < 0)
                sum = 0;
            if (sum > 255)
                sum = 255;

            resultDevice[row*widthResult + col].b = ((char) sum);
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
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
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
