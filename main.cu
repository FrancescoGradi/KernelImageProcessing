#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

__global__ void naiveFiltering(Pixel* pixelsDevice, float* kernelDevice, Pixel* resultDevice, int width, int height,
		int n, int widthResult, int heightResult) {

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((row < heightResult) && (col < widthResult)) {
		float sumR, sumG, sumB;
		int a, b;

		sumR = 0;
		sumG = 0;
		sumB = 0;

		a = 0;

		for (int k = row; k < row + n; k++) {
			b = 0;

			for (int l = col; l < col + n; l++) {
				sumR += kernelDevice[a*n + b] * (int) (unsigned char) pixelsDevice[k*width + l].r;
				sumG += kernelDevice[a*n + b] * (int) (unsigned char) pixelsDevice[k*width + l].g;
				sumB += kernelDevice[a*n + b] * (int) (unsigned char) pixelsDevice[k*width + l].b;

				b++;
			}
			a++;
		}

		if (sumR < 0)
			sumR = 0;
		if (sumR > 255)
			sumR = 255;

		if (sumG < 0)
			sumG = 0;
		if (sumG > 255)
			sumG = 255;

		if (sumB < 0)
			sumB = 0;
		if (sumB > 255)
			sumB = 255;

		resultDevice[row*widthResult + col].r = ((char) sumR);
		resultDevice[row*widthResult + col].g = ((char) sumG);
		resultDevice[row*widthResult + col].b = ((char) sumB);
	}
}

int main() {

	// Parte sequenziale completa
	/*
    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image("images/computer_programming.ppm");

    int n = 3;

    auto* kf = new KernelFactory();

    std::vector<Kernel *> kernels = kf->createAllKernels(n);

    for (auto &kernel : kernels) {
        std::stringstream path;
        path << "images/" << kernel->getType() << n << ".ppm";
        std::string s = path.str();

        (kernel->applyFiltering(img->getPixels(), img->getWidth(),
                img->getHeight(), img->getMagic()))->storeImage(s);
    }

    for (auto &kernel : kernels) {
        delete (kernel);
    }

    kernels.clear();

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    std::cout << "Computation ended after " << duration << " seconds." << std::endl;
    */

	std::cout << "Starting clock..." << std::endl;
	std::clock_t start;

	start = std::clock();
	double duration;

	int n = 3;
	Image* img = new Image("../images/marbles.ppm");

	Pixel* pixels = img->getPixels();
	int width = img->getWidth();
	int height = img->getHeight();

	auto* kf = new KernelFactory();
    std::string filterName = "identity";
	Kernel* kernel = kf->createKernel(n, filterName);

	float* identity = kernel->getFilter();

	int widthResult = width - (n/2) * 2;
	int heightResult = height - (n/2) * 2;

	Pixel* result = new Pixel[widthResult * heightResult];

	// Allocazione memoria nel device
	Pixel* pixelsDevice;
	float* identityDevice;
	Pixel* resultDevice;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(Pixel) * width * height));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * n * n));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(Pixel) * widthResult * heightResult));

	// Copia delle matrici nel device
	CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, n * n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 gridDim(48, 48);
	dim3 blockDim(ceil(((float) widthResult) / gridDim.x), ceil(((float) heightResult) / gridDim.y));

	printf("# pixels totali immagine nuova: %d\n", widthResult * heightResult);

	printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
	printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
	printf("# blocchi: %d\n", gridDim.x * gridDim.y);
	printf("Threads per blocco: %d\n", blockDim.x * blockDim.y);
	printf("Threads totali: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

	// Invocazione del kernel
	naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width, height,
			n, widthResult, heightResult);

	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(Pixel) * widthResult * heightResult,
			cudaMemcpyDeviceToHost));

	Image* newImage = new Image(result, widthResult, heightResult, 255, img->getMagic());

    newImage->storeImage("../images/cuda_" + filterName + ".ppm");

	cudaFree(pixelsDevice);
	cudaFree(identityDevice);
	cudaFree(resultDevice);

	delete [] pixels;
	delete [] identity;

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

	std::cout << "Computation ended after " << duration << " seconds." << std::endl;

    return 0;
}
