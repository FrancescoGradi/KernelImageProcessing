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


double CUDAWithTiling(int kernelSize, std::string imagePath) {

    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
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

    Kernel* kernelI = kf->createKernel(kernelSize, "identity");
    float* identity = kernelI->getFilter();

    Kernel* kernelB = kf->createKernel(kernelSize, "gauss");
    float* blur = kernelB->getFilter();

    Kernel* kernelBB = kf->createKernel(kernelSize, "box");
    float* boxBlur = kernelBB->getFilter();

    Kernel* kernelE = kf->createKernel(kernelSize, "edges");
    float* edge = kernelE->getFilter();

    Kernel* kernelS = kf->createKernel(kernelSize, "sharpen");
    float* sharpen = kernelS->getFilter();

    int widthResult = width - (kernelSize/2) * 2;
    int heightResult = height - (kernelSize/2) * 2;

    float* result = new float[widthResult * heightResult * channels];

    // Allocazione memoria nel device
    float* pixelsDevice;
    float* identityDevice;
    float* blurDevice;
    float* boxBlurDevice;
    float* edgeDevice;
    float* sharpenDevice;
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&blurDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&boxBlurDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edgeDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&sharpenDevice, sizeof(float) * kernelSize * kernelSize));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(blurDevice, blur, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(boxBlurDevice, boxBlur, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edgeDevice, edge, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(sharpenDevice, sharpen, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    // Scelta della dimensione di Grid e di ciascun blocco
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    // Invocazione del kernel per ogni filtro
    tiling<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());
    newImage->storeImage("../images/cuda_tiling_identity.ppm");


    tiling<<<gridDim, blockDim>>>(pixelsDevice, blurDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_tiling_blur.ppm");


    tiling<<<gridDim, blockDim>>>(pixelsDevice, boxBlurDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_tiling_box_blur.ppm");


    tiling<<<gridDim, blockDim>>>(pixelsDevice, edgeDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_tiling_edge.ppm");


    tiling<<<gridDim, blockDim>>>(pixelsDevice, sharpenDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_naive_sharpen.ppm");


    cudaFree(pixelsDevice);
    cudaFree(identityDevice);
    cudaFree(blurDevice);
    cudaFree(boxBlurDevice);
    cudaFree(edgeDevice);
    cudaFree(sharpenDevice);
    cudaFree(resultDevice);

    delete [] pixels;
    delete [] identity;
    delete [] blur;
    delete [] boxBlur;
    delete [] edge;
    delete [] sharpen;
    delete [] result;

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


double CUDANaive(int kernelSize, std::string imagePath) {

    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
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

    Kernel* kernelI = kf->createKernel(kernelSize, "identity");
    float* identity = kernelI->getFilter();

    Kernel* kernelB = kf->createKernel(kernelSize, "gauss");
    float* blur = kernelB->getFilter();

    Kernel* kernelBB = kf->createKernel(kernelSize, "box");
    float* boxBlur = kernelBB->getFilter();

    Kernel* kernelE = kf->createKernel(kernelSize, "edges");
    float* edge = kernelE->getFilter();

    Kernel* kernelS = kf->createKernel(kernelSize, "sharpen");
    float* sharpen = kernelS->getFilter();

    int widthResult = width - (kernelSize/2) * 2;
    int heightResult = height - (kernelSize/2) * 2;

    float* result = new float[widthResult * heightResult * channels];

    // Allocazione memoria nel device
    float* pixelsDevice;
    float* identityDevice;
    float* blurDevice;
    float* boxBlurDevice;
    float* edgeDevice;
    float* sharpenDevice;
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&blurDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&boxBlurDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edgeDevice, sizeof(float) * kernelSize * kernelSize));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&sharpenDevice, sizeof(float) * kernelSize * kernelSize));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(blurDevice, blur, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(boxBlurDevice, boxBlur, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edgeDevice, edge, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(sharpenDevice, sharpen, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    // Scelta della dimensione di Grid e di ciascun blocco
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil(((float) widthResult) / blockDim.x), ceil(((float) heightResult) / blockDim.y));

    // Invocazione del kernel per ogni filtro
    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());
    newImage->storeImage("../images/cuda_naive_identity.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, blurDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_naive_blur.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, boxBlurDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_naive_box_blur.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, edgeDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_naive_edge.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, sharpenDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_naive_sharpen.ppm");


    cudaFree(pixelsDevice);
    cudaFree(identityDevice);
    cudaFree(blurDevice);
    cudaFree(boxBlurDevice);
    cudaFree(edgeDevice);
    cudaFree(sharpenDevice);
    cudaFree(resultDevice);

    delete [] pixels;
    delete [] identity;
    delete [] blur;
    delete [] boxBlur;
    delete [] edge;
    delete [] sharpen;
    delete [] result;

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

double CPPNaive(int kernelSize, std::string imagePath) {
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "C++ sequential naive filtering" << std::endl;
    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image(imagePath);

    auto* kf = new KernelFactory();
    std::vector<Kernel *> kernels = kf->createAllKernels(kernelSize);

    for (auto &kernel : kernels) {
        std::stringstream path;
        path << "../images/sequential_" << kernel->getType() << ".ppm";
        std::string s = path.str();

        (kernel->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getChannels(),
                                img->getMagic()))->storeImage(s);
    }

    for (auto &kernel : kernels) {
        delete(kernel);
    }

    kernels.clear();

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    return duration;

}

double filteringOpenMP(int kernelSize, std::string imagePath, std::string filterName) {
    std::cout << "OpenMP filtering" << std::endl;
    std::cout << "Starting clock..." << std::endl;

    auto start = omp_get_wtime();

    Image* img = new Image(imagePath);

    auto* kf = new KernelFactory();

    Kernel* kernel = kf->createKernel(kernelSize, filterName);

    std::vector<Kernel *> kernels = kf->createAllKernels(kernelSize);
    std::stringstream path;
    path << "../images/openMP_" << kernel->getType() << kernelSize << ".ppm";
    std::string s = path.str();

    (kernel->applyFilteringOpenMP(img->getPixels(), img->getWidth(), img->getHeight(), img->getChannels(),
                            img->getMagic()))->storeImage(s);

    delete kernel;

    kernels.clear();

    double duration = (omp_get_wtime() - start);
    return duration;
}


double filteringOpenMP(int kernelSize, std::string imagePath) {
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "OpenMP filtering" << std::endl;
    std::cout << "Starting clock..." << std::endl;

    auto start = omp_get_wtime();

    Image* img = new Image(imagePath);

    auto* kf = new KernelFactory();
    std::vector<Kernel *> kernels = kf->createAllKernels(kernelSize);

    for (auto &kernel : kernels) {
        std::stringstream path;
        path << "../images/openMP_" << kernel->getType() << ".ppm";
        std::string s = path.str();

        (kernel->applyFilteringOpenMP(img->getPixels(), img->getWidth(), img->getHeight(), img->getChannels(),
                                      img->getMagic()))->storeImage(s);
    }

    for (auto &kernel : kernels) {
        delete(kernel);
    }

    kernels.clear();

    double duration = (omp_get_wtime() - start);
    return duration;

}

// Allocazione memoria nel device

__constant__ float identityDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float blurDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float boxBlurDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float edgeDevice[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float sharpenDevice[KERNEL_SIZE * KERNEL_SIZE];


double CUDAConstantMemory(int kernelSize, std::string imagePath) {

    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
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

    Kernel* kernelI = kf->createKernel(kernelSize, "identity");
    float* identity = kernelI->getFilter();

    Kernel* kernelB = kf->createKernel(kernelSize, "gauss");
    float* blur = kernelB->getFilter();

    Kernel* kernelBB = kf->createKernel(kernelSize, "box");
    float* boxBlur = kernelBB->getFilter();

    Kernel* kernelE = kf->createKernel(kernelSize, "edges");
    float* edge = kernelE->getFilter();

    Kernel* kernelS = kf->createKernel(kernelSize, "sharpen");
    float* sharpen = kernelS->getFilter();

    int widthResult = width - (kernelSize/2) * 2;
    int heightResult = height - (kernelSize/2) * 2;

    float* result = new float[widthResult * heightResult * channels];
    float* pixelsDevice;
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(identityDevice, &identity, kernelSize * kernelSize * sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(blurDevice, &blur, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(boxBlurDevice, &boxBlur, kernelSize * kernelSize * sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edgeDevice, &edge, kernelSize * kernelSize * sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sharpenDevice, &sharpen, kernelSize * kernelSize * sizeof(float), 0, cudaMemcpyHostToDevice));

    // Scelta della dimensione di Grid e di ciascun blocco
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    // Invocazione del kernel per ogni filtro
    // Ho provato a mettere naiveFiltering, ma dà lo stesso errore sia con Tiling che con Constant (che in pratica
    // è tiling senza shared memory, quindi in realtà è abbastanza inutile)
    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());
    newImage->storeImage("../images/cuda_const_identity.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, blurDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_blur.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, boxBlurDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_box_blur.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, edgeDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));;
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_edge.ppm");


    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, sharpenDevice, resultDevice, width,
            height, kernelSize, widthResult, heightResult, channels);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));
    newImage->setPixels(result);
    newImage->storeImage("../images/cuda_const_sharpen.ppm");


    cudaFree(pixelsDevice);
    cudaFree(identityDevice);
    cudaFree(blurDevice);
    cudaFree(boxBlurDevice);
    cudaFree(edgeDevice);
    cudaFree(sharpenDevice);
    cudaFree(resultDevice);

    delete [] pixels;
    delete [] identity;
    delete [] blur;
    delete [] boxBlur;
    delete [] edge;
    delete [] sharpen;
    delete [] result;

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    printf("# pixels totali immagine nuova: %d\n", widthResult * heightResult);
    printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
    printf("# blocchi: %d\n", gridDim.x * gridDim.y);
    printf("Threads per blocco: %d\n", blockDim.x * blockDim.y);
    printf("Threads totali: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    return duration;

}

__constant__ float MASK[KERNEL_SIZE * KERNEL_SIZE];

__global__ void tilingConstant(float* pixelsDevice, float* resultDevice, int width, int height,
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
                sum += N_ds[threadIdx.y + y][threadIdx.x + x] * MASK[y * n + x];
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


double CUDAConstantMemory(int kernelSize, std::string imagePath, std::string filterName) {

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
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(MASK, identity, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    // Invocazione del kernel
    tilingConstant<<<gridDim, blockDim>>>(pixelsDevice, resultDevice, width, height, kernelSize, widthResult, heightResult, channels);

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


