//
// Created by fra on 09/03/19.
//

#include <iostream>
#include "Kernel.h"
#include "Pixel.h"
#include "Image.h"

// Restituisce il kernel piu' semplice, che non modifica l'immagine. Ha un 1 nel mezzo.

Kernel::Kernel(int size, std::string type) {

    this->filter = new float*[size];
    this->size = size;
    this->type = type;

}

Image* Kernel::applyFiltering(Pixel** pixels, int width, int height, std::string magic) {

    // Dopo la convoluzione si riducono le dimensioni dell'immagine,
    width -= (size/2) * 2;
    height -= (size/2) * 2;

    float sumR, sumG, sumB;
    int a, b;

    auto* newPixels = new Pixel*[height];

    for (int i = 0; i < height; i++) {
        newPixels[i] = new Pixel[width];

        for (int j = 0; j < width; j++) {

            sumR = 0;
            sumG = 0;
            sumB = 0;

            a = 0;

            for (int k = i; k < i + size; k++) {
                b = 0;

                for (int l = j; l < j + size; l++) {
                    sumR += filter[a][b] * (int) (unsigned char) pixels[k][l].getR();
                    sumG += filter[a][b] * (int) (unsigned char) pixels[k][l].getG();
                    sumB += filter[a][b] * (int) (unsigned char) pixels[k][l].getB();

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

            newPixels[i][j].setR((char) sumR);
            newPixels[i][j].setG((char) sumG);
            newPixels[i][j].setB((char) sumB);
        }
    }

    return new Image(newPixels, width, height, 255, magic);

}

// TODO: trovare il modo di evitare il segmentation fault che si ha quando il filtro non esiste
float** Kernel::getFilter() {

    if (this->filter != nullptr) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                std::cout << this->filter[i][j] << " ";
            }
            std::cout << "" << std::endl;
        }
        return this->filter;
    }
    else {
        return nullptr;
    }
}

Kernel::~Kernel() {

    if (filter != nullptr) {
        for (int i = 0; i < size; i++) {
            delete [] filter[i];
        }

        delete [] filter;
    }
}

std::string Kernel::getType() {
    return this->type;
}
