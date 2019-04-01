//
// Created by fra on 09/03/19.
//

#include "Kernel.h"
#include "Image.h"

#include <cmath>
#include <iostream>
#include <omp.h>

Kernel::Kernel(std::string type) {

    this->size = 3;
    this->filter = new float[size * size];
    this->type = type;

}
Kernel::Kernel(int size, std::string type) {

    this->filter = new float[size * size];
    this->size = size;
    this->type = type;

}

Image* Kernel::applyFiltering(float* pixels, int width, int height, int channels, std::string magic) {

    int oldWidth = width;

    width -= (size/2) * 2;
    height -= (size/2) * 2;

    float sum;
    int a, b;

    auto* newPixels = new float[height * width * channels];

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {

                sum = 0;
                a = 0;

                for (int k = i; k < i + size; k++) {
                    b = 0;

                    for (int l = j; l < j + size; l++) {
                        sum += filter[a * size + b] * pixels[k * oldWidth * channels + l * channels + c];
                        b++;
                    }
                    a++;
                }

                if (sum < 0)
                    sum = 0;
                if (sum > 1)
                    sum = 1;

                newPixels[i * width * channels + j * channels + c] = sum;
            }
        }
    }

    return new Image(newPixels, width, height, 255, channels, magic);

}

Image* Kernel::applyFilteringOpenMP(float* pixels, int width, int height, int channels, std::string magic) {

    int oldWidth = width;

    width -= (size/2) * 2;
    height -= (size/2) * 2;

    float sum;
    int a, b;

    auto* newPixels = new float[height * width * channels];

#pragma omp parallel for private(sum, a, b) schedule(static, 1024) num_threads(42)
    for (int i = 0; i < height; i++) {
        for (int c = 0; c < channels; ++c) {
            for (int j = 0; j < width; j++) {

                sum = 0;
                a = 0;

                for (int k = i; k < i + size; k++) {
                    b = 0;

                    for (int l = j; l < j + size; l++) {
                        sum += filter[a * size + b] * pixels[k * oldWidth * channels + l * channels + c];
                        b++;
                    }
                    a++;
                }

                if (sum < 0)
                    sum = 0;
                if (sum > 1)
                    sum = 1;

                newPixels[i * width * channels + j * channels + c] = sum;
            }
        }
    }

    return new Image(newPixels, width, height, 255, channels, magic);

}

float* Kernel::getFilter() {

    if (this->filter != nullptr)
        return this->filter;
    else
        return nullptr;
}

Kernel::~Kernel() {

    if (filter != nullptr)
        delete [] filter;
}

std::string Kernel::getType() {
    return this->type;
}
