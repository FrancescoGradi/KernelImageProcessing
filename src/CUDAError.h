#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <curand_mtgp32_kernel.h>

#ifndef KERNELIMAGEPROCESSING_CUDAERROR_H
#define KERNELIMAGEPROCESSING_CUDAERROR_H

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

static void CheckCudaErrorAux(const char *file, unsigned line,
                              const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#endif //KERNELIMAGEPROCESSING_CUDAERROR_H
