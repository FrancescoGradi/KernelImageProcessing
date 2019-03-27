//
// Created by federico on 10/03/19.
//

#include <iostream>
#include "KernelFactory.h"
#include "Filters/GaussianBlur.h"
#include "Filters/Identity.h"
#include "Filters/Sharpen.h"
#include "Filters/BoxBlur.h"
#include "Filters/EdgeDetection.h"

Kernel* KernelFactory::createKernel(int size, std::string type) {

    Kernel* filter;

    if (type == "identity") {
        filter = new Identity(type, size);
    }
    else if (type == "gauss") {
        filter = new GaussianBlur(type, size);
    }
    else if (type == "sharpen") {
        filter = new Sharpen(type);
    }
    else if (type == "box") {
        filter = new BoxBlur(type, size);
    }
    else if (type == "edges") {
        filter = new EdgeDetection(type);
    }
    else {
        std::cout << "Filter type not known." << std::endl;
        filter = nullptr;
    }
    return filter;

}

std::vector<Kernel *> KernelFactory::createAllKernels(int size) {

    std::vector<Kernel *> kernels;

    kernels.push_back(new Identity("identity", size));
    kernels.push_back(new GaussianBlur("blur", size));
    kernels.push_back(new Sharpen("sharpen"));
    kernels.push_back(new BoxBlur("boxBlur", size));
    kernels.push_back(new EdgeDetection("edge"));

    return kernels;
}
