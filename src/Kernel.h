//
// Created by fra on 09/03/19.
//

#ifndef KERNELIMAGEPROCESSING_KERNEL_H
#define KERNELIMAGEPROCESSING_KERNEL_H


#include "Image.h"

class Kernel {

public:

    explicit Kernel(std::string type);
    Kernel(int size, std::string type);
    virtual ~Kernel();

    std::string getType();

    float* getFilter();

    Image* applyFiltering(float* pixels, int width, int height, int channels, std::string magic);
    Image *applyFilteringOpenMP(float *pixels, int width, int height, int channels, std::string magic);

protected:
    float* filter;
    int size;

private:
    std::string type;
};


#endif //KERNELIMAGEPROCESSING_KERNEL_H
