#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

int main() {

    Image* img = new Image("../images/computer_programming.ppm");

    int n = 3;

    auto* kf = new KernelFactory();

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

    return 0;

}