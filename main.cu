#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <curand_mtgp32_kernel.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define TILE_WIDTH 8
#define w (TILE_WIDTH + 3 - 1)


static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

__global__ void naiveFiltering(float* pixelsDevice, float* kernelDevice, float* resultDevice, int width, int height,
		int n, int widthResult, int heightResult, int channels) {

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
				sumR += kernelDevice[a*n + b] * pixelsDevice[k*width*channels + l*channels + 0];
				sumG += kernelDevice[a*n + b] * pixelsDevice[k*width*channels + l*channels + 1];
				sumB += kernelDevice[a*n + b] * pixelsDevice[k*width*channels + l*channels + 2];

				b++;
			}
			a++;
		}

		if (sumR < 0)
			sumR = 0;
		if (sumR > 1)
			sumR = 1;

		if (sumG < 0)
			sumG = 0;
		if (sumG > 1)
			sumG = 1;

		if (sumB < 0)
			sumB = 0;
		if (sumB > 1)
			sumB = 1;

		resultDevice[row*widthResult*channels + col*channels + 0] = sumR;
		resultDevice[row*widthResult*channels + col*channels + 1] = sumG;
		resultDevice[row*widthResult*channels + col*channels + 2] = sumB;
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


int main() {

	// Parte sequenziale completa
    /*
    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image("../images/computer_programming.ppm");

    int n = 3;

    auto* kf = new KernelFactory();

    std::vector<Kernel *> kernels = kf->createAllKernels(n);

    for (auto &kernel : kernels) {
        std::stringstream path;
        path << "../images/" << kernel->getType() << n << ".ppm";
        std::string s = path.str();

        (kernel->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getChannels(),
                img->getMagic()))->storeImage(s);
    }

    for (auto &kernel : kernels) {
        delete (kernel);
    }

    kernels.clear();

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    std::cout << "Computation ended after " << duration << " seconds." << std::endl;


    // Parte parallela CUDA senza Tiling

	std::cout << "Starting clock..." << std::endl;
	std::clock_t start;

	start = std::clock();
	double duration;

	int n = 3;
	Image* img = new Image("../images/marbles.ppm");

	float* pixels = img->getPixels();
	int width = img->getWidth();
	int height = img->getHeight();
	int channels = img->getChannels();

	auto* kf = new KernelFactory();
    std::string filterName = "identity";
	Kernel* kernel = kf->createKernel(n, filterName);

	float* identity = kernel->getFilter();

	int widthResult = width - (n/2) * 2;
	int heightResult = height - (n/2) * 2;

	float* result = new float[widthResult * heightResult * channels];

	// Allocazione memoria nel device
	float* pixelsDevice;
	float* identityDevice;
	float* resultDevice;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * n * n));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

	// Copia delle matrici nel device
	CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, n * n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockDim(32, 32);
    dim3 gridDim(ceil(((float) widthResult) / blockDim.x), ceil(((float) heightResult) / blockDim.y));

	printf("# pixels totali immagine nuova: %d\n", widthResult * heightResult);

	printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
	printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
	printf("# blocchi: %d\n", gridDim.x * gridDim.y);
	printf("Threads per blocco: %d\n", blockDim.x * blockDim.y);
	printf("Threads totali: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    // Invocazione del kernel
    naiveFiltering<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, n, widthResult, heightResult, channels);

	cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));

	Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());

    newImage->storeImage("../images/cuda_" + filterName + ".ppm");

	cudaFree(pixelsDevice);
	cudaFree(identityDevice);
	cudaFree(resultDevice);

	delete [] pixels;
	delete [] identity;

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

	std::cout << "Computation ended after " << duration << " seconds." << std::endl;

    */

    // Codice Cuda con Tiling

    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    int n = 3;
    Image* img = new Image("../images/computer_programming.ppm");

    float* pixels = img->getPixels();
    int width = img->getWidth();
    int height = img->getHeight();
    int channels = img->getChannels();

    auto* kf = new KernelFactory();
    std::string filterName = "identity";
    Kernel* kernel = kf->createKernel(n, filterName);

    float* identity = kernel->getFilter();

    int widthResult = width - (n/2) * 2;
    int heightResult = height - (n/2) * 2;

    float* result = new float[widthResult * heightResult * channels];

    // Allocazione memoria nel device
    float* pixelsDevice;
    float* identityDevice;
    float* resultDevice;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&pixelsDevice, sizeof(float) * width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&identityDevice, sizeof(float) * n * n));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&resultDevice, sizeof(float) * widthResult * heightResult * channels));

    // Copia delle matrici nel device
    CUDA_CHECK_RETURN(cudaMemcpy(pixelsDevice, pixels, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(identityDevice, identity, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(((float) widthResult) / TILE_WIDTH), ceil(((float) heightResult) / TILE_WIDTH));

    printf("# pixels totali immagine nuova: %d\n", widthResult * heightResult);

    printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
    printf("# blocchi: %d\n", gridDim.x * gridDim.y);
    printf("Threads per blocco: %d\n", blockDim.x * blockDim.y);
    printf("Threads totali: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    // Invocazione del kernel
    tiling<<<gridDim, blockDim>>>(pixelsDevice, identityDevice, resultDevice, width,
            height, n, widthResult, heightResult, channels);

    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(result, resultDevice, sizeof(float) * widthResult * heightResult * channels,
                                 cudaMemcpyDeviceToHost));

    Image* newImage = new Image(result, widthResult, heightResult, 255, channels, img->getMagic());

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
