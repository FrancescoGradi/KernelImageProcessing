#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
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

	if ((row < widthResult) && (col < heightResult)) {
		float sumR, sumG, sumB;
		int a, b;

		sumR = 0;
		sumG = 0;
		sumB = 0;

		a = 0;

		for (int k = row; k < row + n; k++) {
			b = 0;

			for (int l = col; l < col + n; l++) {
				sumR += kernelDevice[a*n + b] * (int) (unsigned char) pixelsDevice[k*width + l].getR();
				sumG += kernelDevice[a*n + b] * (int) (unsigned char) pixelsDevice[k*width + l].getG();
				sumB += kernelDevice[a*n + b] * (int) (unsigned char) pixelsDevice[k*width + l].getB();

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

		resultDevice[row*widthResult + col].setR((char) sumR);
		resultDevice[row*widthResult + col].setG((char) sumG);
		resultDevice[row*widthResult + col].setB((char) sumB);
	}
}

int main() {

	// Parte sequenziale completa

    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image("images/computer_programming.ppm");

    int n = 5;

    auto* kf = new KernelFactory();

    // TODO aggiornare i filtri con matrice unidimensionale. Per quanto riguarda i new sembrerebbe che a CUDA vadano
    //  bene, in ogni caso e' veloce mettere malloc al posto dei new
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
    /*

	int n = 5;
	Image* img = new Image("images/computer_programming.ppm");

	Pixel* pixels = img->getPixels();
	int width = img->getWidth();
	int height = img->getHeight();

	auto* kf = new KernelFactory();
	Kernel* kernel = kf->createKernel(n, "identity");

	float* identity = kernel->getFilter();

	int widthResult = width - (n/2) * 2;
	int heightResult = height - (n/2) * 2;
	Pixel* result = new Pixel[widthResult * heightResult];

	// Allocazione memoria nel device
	Pixel* pixelsDevice;
	float* identityDevice;
	Pixel* resultDevice;

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pixelsDevice, sizeof(Pixel) * width * height));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * n * n));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(Pixel) * widthResult * heightResult));

	// Copia delle matrici nel device
	CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, n * n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockDim = (n, n);
	dim3 gridDim = (widthResult, heightResult);

	// Invocazione del kernel
	naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width, height,
			n, widthResult, heightResult);

	cudaDeviceSynchronize();

	cudaFree(pixelsDevice);
	cudaFree(identityDevice);
	cudaFree(resultDevice);

	delete [] pixels;
	delete [] identity;
	delete [] result;
	*/

    return 0;

}
