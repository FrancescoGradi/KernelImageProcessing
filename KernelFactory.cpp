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
        std::cout << "Creating identity filter..." << std::endl;
        filter = new Identity(type, size);
    }
    else if (type == "gauss") {
        std::cout << "Creating gaussian blur filter..." << std::endl;
        filter = new GaussianBlur(type, size);
    }
    else if (type == "sharpen") {
        std::cout << "Creating sharpening filter..." << std::endl;
        filter = new Sharpen(type, size);
    }
    else if (type == "box") {
        std::cout << "Creating box blur filter..." << std::endl;
        filter = new BoxBlur(type, size);
    }
    else if (type == "edges") {
        std::cout << "Creating edge detector filter..." << std::endl;
        filter = new EdgeDetection(type, size);
    }
    else {
        std::cout << "Filter type not known." << std::endl;
        filter = nullptr;
    }
    return filter;

}