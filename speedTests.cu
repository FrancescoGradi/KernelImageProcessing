#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "CUDAError.h"
#include "filteringUtils.h"

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

double CUDAWithTiling(int kernelSize, std::string imagePath, std::string filterName) {

    std::cout << "CUDA tiling filtering" << std::endl;
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
    float* identityDevice;
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    // Invocazione del kernel
    tiling<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);

    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));

    Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());

    newImage->storeImage("../images/cuda_tiling_" + filterName + ".ppm");

    cudaFree(pixelsDevice);
    cudaFree(identityDevice);
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

double CUDANaive(int kernelSize, std::string imagePath, std::string filterName) {

    std::cout << "CUDA naive filtering" << std::endl;
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
	float* identityDevice;
	float* resultDevice;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * kernelSize * kernelSize));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

	// Copia delle matrici nel device
	CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockDim(32, 32);
    dim3 gridDim(ceil(((float) widthResult) / blockDim.x), ceil(((float) heightResult) / blockDim.y));

    // Invocazione del kernel
    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);

	cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));

	Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());

    newImage->storeImage("../images/cuda_naive_" + filterName + ".ppm");

	cudaFree(pixelsDevice);
	cudaFree(identityDevice);
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

double CPPNaive(int kernelSize, std::string imagePath, std::string filterName) {

    std::cout << "C++ sequential naive filtering" << std::endl;
    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image(imagePath);

    auto* kf = new KernelFactory();

    Kernel* kernel = kf->createKernel(kernelSize, filterName);

    std::vector<Kernel *> kernels = kf->createAllKernels(kernelSize);
    std::stringstream path;
    path << "../images/" << kernel->getType() << kernelSize << ".ppm";
    std::string s = path.str();

    (kernel->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getChannels(),
                           img->getMagic()))->storeImage(s);

    delete kernel;

    kernels.clear();

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    return duration;

}

