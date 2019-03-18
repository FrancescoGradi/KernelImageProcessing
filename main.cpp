#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>

int main() {

    std::cout << "Starting clock..." << std::endl;
    std::clock_t start;

    start = std::clock();
    double duration;

    Image* img = new Image("../images/computer_programming.ppm");

    int n = 3;

    auto* kf = new KernelFactory();

    // TODO aggiornare i filtri con matrice unidimensionale. Per quanto riguarda i new sembrerebbe che a CUDA vadano
    //  bene, in ogni caso e' veloce mettere malloc al posto dei new
    std::vector<Kernel *> kernels = kf->createAllKernels(n);

    for (auto &kernel : kernels) {
        std::stringstream path;
        path << "../images/" << kernel->getType() << n << ".ppm";
        std::string s = path.str();

        (kernel->applyFiltering(img->getPixels(), img->getWidth(),
                img->getHeight(), img->getMagic()))->storeImage(s);
    }

    for (auto &kernel : kernels) {
        delete (kernel);
    }

    kernels.clear();

    duration = (std::clock() - start) / double CLOCKS_PER_SEC;

    std::cout << "Computation ended after " << duration << " seconds." << std::endl;

    return 0;

}