//
// Created by federico on 10/03/19.
//

#ifndef KERNELIMAGEPROCESSING_KERNELFACTORY_H
#define KERNELIMAGEPROCESSING_KERNELFACTORY_H


#include "Kernel.h"

#include <vector>

class KernelFactory {

public:

    Kernel* createKernel(int size, std::string type);
    std::vector<Kernel*> createAllKernels(int size);

};


#endif //KERNELIMAGEPROCESSING_KERNELFACTORY_H
