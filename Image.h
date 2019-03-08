//
// Created by fra on 06/03/19.
//

#ifndef KERNELIMAGEPROCESSING_IMAGE_H
#define KERNELIMAGEPROCESSING_IMAGE_H


#include "Pixel.h"

class Image {

public:
    Image();
    ~Image() {}

private:
    int width;
    int height;
    Pixel** pixels;
};


#endif //KERNELIMAGEPROCESSING_IMAGE_H
