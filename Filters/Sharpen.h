//
// Created by federico on 11/03/19.
//

#ifndef KERNELIMAGEPROCESSING_SHARPEN_H
#define KERNELIMAGEPROCESSING_SHARPEN_H


#include "../Kernel.h"

class Sharpen: public Kernel {

public:
    Sharpen(std::string type, int size);

};


#endif //KERNELIMAGEPROCESSING_SHARPEN_H
