//
// Created by federico on 10/03/19.
//

#ifndef KERNELIMAGEPROCESSING_GAUSSIANBLUR_H
#define KERNELIMAGEPROCESSING_GAUSSIANBLUR_H


#include "../Kernel.h"

class GaussianBlur: public Kernel {

public:
    GaussianBlur(std::string type, int size);

};


#endif //KERNELIMAGEPROCESSING_GAUSSIANBLUR_H
