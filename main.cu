#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "speedTests.h"

int main() {

    double durationCUDATiling = CUDAWithTiling(3, "../images/marbles.ppm", "sharpen");
    std::cout << "Computation ended after " << durationCUDATiling << " seconds." << std::endl;
    std::cout << "" << std::endl;
    double durationCUDANaive = CUDANaive(3, "../images/marbles.ppm", "sharpen");
    std::cout << "Computation ended after " << durationCUDANaive << " seconds." << std::endl;
    std::cout << "" << std::endl;
    double durationCPPNaive = CPPNaive(3, "../images/marbles.ppm", "sharpen");
    std::cout << "Computation ended after " << durationCPPNaive << " seconds." << std::endl;
    std::cout << "" << std::endl;

    return 0;
}
