#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "speedTests.h"

int main() {

    std::string path = "../images/computer_programming.ppm";

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

    return 0;
}
