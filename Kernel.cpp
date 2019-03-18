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

Image* Kernel::applyFiltering(Pixel* pixels, int width, int height, std::string magic) {

    // Dopo la convoluzione si riducono le dimensioni dell'immagine,
    width -= (size/2) * 2;
    height -= (size/2) * 2;

    float sumR, sumG, sumB;
    int a, b;

    auto* newPixels = new Pixel[height * width];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            sumR = 0;
            sumG = 0;
            sumB = 0;

            a = 0;

            for (int k = i; k < i + size; k++) {
                b = 0;

                for (int l = j; l < j + size; l++) {

                    // TODO sembrerebbe che l'errore sia negli indici su k e l qua...
                    sumR += filter[a*size + b] * (int) (unsigned char) pixels[k*width + l].getR();
                    sumG += filter[a*size + b] * (int) (unsigned char) pixels[k*width + l].getG();
                    sumB += filter[a*size + b] * (int) (unsigned char) pixels[k*width + l].getB();

                    b++;
                }
                a++;
            }

            if (sumR < 0)
                sumR = 0;
            if (sumR > 255)
                sumR = 255;

            if (sumG < 0)
                sumG = 0;
            if (sumG > 255)
                sumG = 255;

            if (sumB < 0)
                sumB = 0;
            if (sumB > 255)
                sumB = 255;

            newPixels[i*width + j].setR((char) sumR);
            newPixels[i*width + j].setG((char) sumG);
            newPixels[i*width + j].setB((char) sumB);
        }
    }

    return new Image(newPixels, width, height, 255, magic);

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
