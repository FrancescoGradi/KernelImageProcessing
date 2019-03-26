#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "speedTests.h"

int main() {

    std::string path = "../images/milky.ppm";
    int kernelSize = 3;

    double durationCUDATiling = CUDAWithTiling(kernelSize, path, "edges");
    std::cout << "Computation ended after " << durationCUDATiling << " seconds." << std::endl;
    std::cout << "" << std::endl;
    double durationCUDANaive = CUDANaive(kernelSize, path, "edges");
    std::cout << "Computation ended after " << durationCUDANaive << " seconds." << std::endl;
    std::cout << "" << std::endl;
    double durationCPPNaive = CPPNaive(kernelSize, path, "edges");
    std::cout << "Computation ended after " << durationCPPNaive << " seconds." << std::endl;
    std::cout << "" << std::endl;

    return 0;
}
