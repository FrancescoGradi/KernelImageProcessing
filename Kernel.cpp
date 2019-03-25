//
// Created by fra on 09/03/19.
//

#include "Kernel.h"
#include "Pixel.h"
#include "Image.h"

#include <cmath>

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

    // TODO fare un ciclo for esterno per i canali, cosi' da evitare il ripetersi della stessa roba

    // Dopo la convoluzione si riducono le dimensioni dell'immagine,
    int oldWidth = width;

    width -= (size/2) * 2;
    height -= (size/2) * 2;

    float sumR, sumG, sumB;
    int a, b;

    auto* newPixels = new float[height * width * channels];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            sumR = 0;
            sumG = 0;
            sumB = 0;

            a = 0;

            for (int k = i; k < i + size; k++) {
                b = 0;

                for (int l = j; l < j + size; l++) {
                    sumR += filter[a*size + b] * pixels[k*oldWidth*channels + l*channels + 0];
                    sumG += filter[a*size + b] * pixels[k*oldWidth*channels + l*channels + 1];
                    sumB += filter[a*size + b] * pixels[k*oldWidth*channels + l*channels + 2];

                    b++;
                }
                a++;
            }

            if (sumR < 0)
                sumR = 0;
            if (sumR > 1)
                sumR = 1;

            if (sumG < 0)
                sumG = 0;
            if (sumG > 1)
                sumG = 1;

            if (sumB < 0)
                sumB = 0;
            if (sumB > 1)
                sumB = 1;

            newPixels[i*width*channels + j*channels + 0] = sumR;
            newPixels[i*width*channels + j*channels + 1] = sumG;
            newPixels[i*width*channels + j*channels + 2] = sumB;
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
