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

    std::string path = "../images/original/ridridSunset.ppm";

    std::cout << "" << std::endl;
    std::cout << "Single Tests: a single filtering for each convolution method." << std::endl;

    double durationCPPNaive = CPPNaive(3, path, "gauss");
    std::cout << "C++ naive: " << durationCPPNaive << std::endl;

    double durationOpenMP = filteringOpenMP(3, path, "gauss");
    std::cout << "OpenMP: " << durationOpenMP << "   Speed up: " << durationCPPNaive/durationOpenMP << std::endl;

    double durationCUDANaive = CUDANaive(3, path, "gauss");
    std::cout << "CUDA naive: " << durationCUDANaive << "   Speed up: " << durationCPPNaive/durationCUDANaive << std::endl;

    double durationCUDAConstant = CUDAConstantMemory(3, path, "gauss");
    std::cout << "CUDA constant memory: " << durationCUDAConstant << "   Speed up: " << durationCPPNaive/durationCUDAConstant << std::endl;

    double durationCUDATiling = CUDAWithTiling(3, path, "gauss");
    std::cout << "CUDA tiling: " << durationCUDATiling << "   Speed up: " << durationCPPNaive/durationCUDATiling << std::endl;


    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Mutiple Tests: all mask filtering for each convolution method." << std::endl;

    durationCPPNaive = CPPNaive(3, path);
    std::cout << "C++ naive: " << durationCPPNaive << std::endl;

    durationOpenMP = filteringOpenMP(3, path);
    std::cout << "OpenMP: " << durationOpenMP << "   Speed up: " << durationCPPNaive/durationOpenMP << std::endl;

    durationCUDANaive = CUDANaive(3, path);
    std::cout << "CUDA naive: " << durationCUDANaive << "   Speed up: " << durationCPPNaive/durationCUDANaive << std::endl;

    durationCUDATiling = CUDAWithTiling(3, path);
    std::cout << "CUDA tiling: " << durationCUDATiling << "   Speed up: " << durationCPPNaive/durationCUDATiling << std::endl;

    durationCUDAConstant = CUDAConstantMemory(3, path);
    std::cout << "CUDA constant memory: " << durationCUDAConstant << "   Speed up: " << durationCPPNaive/durationCUDAConstant << std::endl;


    return 0;
}
