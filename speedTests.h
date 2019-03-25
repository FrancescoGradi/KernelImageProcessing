//
// Created by federico on 25/03/19.
//

#ifndef KERNELIMAGEPROCESSING_SPEEDTESTS_H
#define KERNELIMAGEPROCESSING_SPEEDTESTS_H

#include <iostream>

double CUDAWithTiling(int kernelSize, std::string imagePath, std::string filterName);
double CUDANaive(int kernelSize, std::string imagePath, std::string filterName);
double CPPNaive(int kernelSize, std::string imagePath, std::string filterName);

#endif //KERNELIMAGEPROCESSING_SPEEDTESTS_H