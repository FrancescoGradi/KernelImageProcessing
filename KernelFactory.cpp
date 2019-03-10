//
// Created by federico on 10/03/19.
//

#include <iostream>
#include "KernelFactory.h"
#include "Filters/GaussianBlur.h"
#include "Filters/Identity.h"

Kernel* KernelFactory::createKernel(int size, std::string type) {

    Kernel* filter;

    if (type == "identity") {
        std::cout << "Creating identity filter..." << std::endl;
        filter = new Identity(type, size);
    }
    else if (type == "gauss") {
        std::cout << "Creating gaussian blur filter..." << std::endl;
        filter = new GaussianBlur(type, size);
    }
    else {
        std::cout << "Filter type not known." << std::endl;
        filter = nullptr;
    }
    return filter;

}