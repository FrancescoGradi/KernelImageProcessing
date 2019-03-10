//
// Created by federico on 10/03/19.
//

#ifndef KERNELIMAGEPROCESSING_KERNELFACTORY_H
#define KERNELIMAGEPROCESSING_KERNELFACTORY_H


#include "Kernel.h"

class KernelFactory {

public:

    Kernel* createKernel(int size, std::string type);

};


#endif //KERNELIMAGEPROCESSING_KERNELFACTORY_H
