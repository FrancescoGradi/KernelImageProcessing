#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "CUDAError.h"
#include "filteringUtils.h"
#include "speedTests.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <curand_mtgp32_kernel.h>
#include <omp.h>

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define TILE_WIDTH 32
#define w (TILE_WIDTH + 3 - 1)
#define KERNEL_SIZE 3

__constant__ float MASK[KERNEL_SIZE * KERNEL_SIZE];

__constant__ float identityDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float blurDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float boxBlurDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float edgeDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float sharpenDevice[KERNEL_SIZE * KERNEL_SIZE];

__global__ void constantMemoryFiltering(float* pixelsDevice, float* resultDevice, int width, int height,
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
                    sum += MASK[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
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

__global__ void constantMemoryFilteringIdentity(float* pixelsDevice, float* resultDevice, int width, int height,
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
                    sum += identityDevice[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
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

__global__ void constantMemoryFilteringBlur(float* pixelsDevice, float* resultDevice, int width, int height,
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
                    sum += blurDevice[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
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

__global__ void constantMemoryFilteringBoxBlur(float* pixelsDevice, float* resultDevice, int width, int height,
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
                    sum += boxBlurDevice[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
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

__global__ void constantMemoryFilteringEdge(float* pixelsDevice, float* resultDevice, int width, int height,
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
                    sum += edgeDevice[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
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

__global__ void constantMemoryFilteringSharpen(float* pixelsDevice, float* resultDevice, int width, int height,
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
                    sum += sharpenDevice[a * n + b] * pixelsDevice[k * width * channels + l * channels + i];
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

double CUDAConstantMemory(int kernelSize, std::string imagePath, std::string filterName) {

    std::cout << "CUDA constant memory filtering" << std::endl;
    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image(imagePath);

    float* pixels = img->getPixels();
    int width = img->getWidth();
    int height = img->getHeight();
    int channels = img->getChannels();

    auto* kf = new KernelFactory();
    Kernel* kernel = kf->createKernel(kernelSize, filterName);

    float* identity = kernel->getFilter();

    int widthResult = width - (kernelSize/2) * 2;
    int heightResult = height - (kernelSize/2) * 2;

    float* result = new float[widthResult * heightResult * channels];

    // Allocazione memoria nel device
    float* pixelsDevice;
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(MASK, identity, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    // Invocazione del kernel
    constantMemoryFiltering<<<gridDim, blockDim>>>(pixelsDevice, resultDevice, width, height, kernelSize, widthResult, heightResult, channels);

    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));

    Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());

    newImage->storeImage("../images/cuda_constant_" + filterName + ".ppm");

    cudaFree(pixelsDevice);
    cudaFree(resultDevice);

    delete [] pixels;
    delete [] identity;

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    printf("# pixels totali immagine nuova: %d\n", widthResult * heightResult);
    printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
    printf("# blocchi: %d\n", gridDim.x * gridDim.y);
    printf("Threads per blocco: %d\n", blockDim.x * blockDim.y);
    printf("Threads totali: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    return duration;
}



double CUDAConstantMemory(int kernelSize, std::string imagePath) {

    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "CUDA constant memory filtering" << std::endl;
    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image *img = new Image(imagePath);

    float *pixels = img->getPixels();
    int width = img->getWidth();
    int height = img->getHeight();
    int channels = img->getChannels();

    auto *kf = new KernelFactory();

    Kernel *kernelI = kf->createKernel(kernelSize, "identity");
    float *identity = kernelI->getFilter();

    Kernel *kernelB = kf->createKernel(kernelSize, "gauss");
    float *blur = kernelB->getFilter();

    Kernel *kernelBB = kf->createKernel(kernelSize, "box");
    float *boxBlur = kernelBB->getFilter();

    Kernel *kernelE = kf->createKernel(kernelSize, "edges");
    float *edge = kernelE->getFilter();

    Kernel *kernelS = kf->createKernel(kernelSize, "sharpen");
    float *sharpen = kernelS->getFilter();

    int widthResult = width - (kernelSize / 2) * 2;
    int heightResult = height - (kernelSize / 2) * 2;

    float *result = new float[widthResult * heightResult * channels];
    float *pixelsDevice;
    float *resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &pixelsDevice, sizeof(float) * width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(identityDevice, identity, kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(blurDevice, blur, kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(boxBlurDevice, boxBlur, kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edgeDevice, edge, kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sharpenDevice, sharpen, kernelSize * kernelSize * sizeof(float)));

    // Scelta della dimensione di Grid e di ciascun blocco
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    // Invocazione del kernel per ogni filtro

    constantMemoryFilteringIdentity<<< gridDim, blockDim >>> (pixelsDevice, resultDevice, width, height, kernelSize,
            widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    Image *newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());
    newImage->storeImage("../images/cuda_const_identity.ppm");


    constantMemoryFilteringBlur<<< gridDim, blockDim >>> (pixelsDevice, resultDevice, width, height, kernelSize,
            widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_blur.ppm");


    constantMemoryFilteringBoxBlur<<< gridDim, blockDim >>> (pixelsDevice, resultDevice, width, height, kernelSize,
            widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_box_blur.ppm");


    constantMemoryFilteringEdge<<< gridDim, blockDim >>> (pixelsDevice, resultDevice, width, height, kernelSize,
            widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_edge.ppm");


    constantMemoryFilteringSharpen<<< gridDim, blockDim >>> (pixelsDevice, resultDevice, width, height, kernelSize,
            widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_sharpen.ppm");

    cudaFree(pixelsDevice);
    cudaFree(resultDevice);

    delete[] pixels;
    delete[] identity;
    delete[] blur;
    delete[] boxBlur;
    delete[] edge;
    delete[] sharpen;
    delete[] result;

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    printf("# pixels totali immagine nuova: %d\n", widthResult * heightResult);
    printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
    printf("# blocchi: %d\n", gridDim.x * gridDim.y);
    printf("Threads per blocco: %d\n", blockDim.x * blockDim.y);
    printf("Threads totali: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    return duration;

}