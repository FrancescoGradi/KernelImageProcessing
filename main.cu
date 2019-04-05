#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "speedTests.h"
#include "constantMemoryUtils.h"

#include <chrono>
#include <omp.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#ifndef w
#define w (TILE_WIDTH + 3 - 1)
#endif

int main() {

    std::string path = "../images/original/milky.ppm";

    /*

    std::string kernels[5] = {"identity", "gauss", "box", "edges", "sharpen"};

    double durationCUDATiling[5];
    double durationCUDANaive[5];
    double durationCPPNaive[5];

    for(int i = 0; i < 5; i++) {
        durationCUDATiling[i] = CUDAWithTiling(3, path, kernels[i]);
        durationCUDANaive[i] = CUDANaive(3, path, kernels[i]);
        durationCPPNaive[i] = CPPNaive(3, path, kernels[i]);
    }

    for(int i = 0; i < 5; i++) {
        std::cout << kernels[i] << " duration: CUDA with tiling: " << durationCUDATiling[i];
        std::cout << " | CUDA naive: " << durationCUDANaive[i];
        std::cout << " | C++ naive: " << durationCPPNaive[i] << std::endl;
    }

    */

    double durationCPPNaive = CPPNaive(3, path);
    std::cout << "C++ naive: " << durationCPPNaive << std::endl;

    double durationOpenMP = filteringOpenMP(3, path);
    std::cout << "OpenMP: " << durationOpenMP << "   Speed up: " << durationCPPNaive/durationOpenMP << std::endl;

    double durationCUDANaive = CUDANaive(3, path);
    std::cout << "CUDA naive: " << durationCUDANaive << "   Speed up: " << durationCPPNaive/durationCUDANaive << std::endl;

    double durationCUDATiling = CUDAWithTiling(3, path);
    std::cout << "CUDA tiling: " << durationCUDATiling << "   Speed up: " << durationCPPNaive/durationCUDATiling << std::endl;

    double durationCUDAConstant = CUDAConstantMemory(3, path);
    std::cout << "CUDA constant memory: " << durationCUDAConstant << "   Speed up: " << durationCPPNaive/durationCUDAConstant << std::endl;


    return 0;
}
