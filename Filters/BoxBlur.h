//
// Created by federico on 11/03/19.
//

#ifndef KERNELIMAGEPROCESSING_BOXBLUR_H
#define KERNELIMAGEPROCESSING_BOXBLUR_H


#include "../Kernel.h"

class BoxBlur: public Kernel {

public:
    BoxBlur(std::string type, int size);

};


#endif //KERNELIMAGEPROCESSING_BOXBLUR_H
