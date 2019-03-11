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

Image* Kernel::applyFiltering(Pixel** pixels, float** filter, int width, int height, int size) {

    auto* newPixels = new float* [width];

    for (int i = size/2; i < height; i++) {
        newPixels[i] = new float[height];
        for (int j = size/2; j < width; j++) {
            // si moltiplicano i valori dell'immagine per i valori del filtro e si calcola il nuovo valore, si inserisce
            // in una nuova immagine e si rende quella


        }
    }

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

    for (int i = 0; i < size; i++) {
        delete [] filter[i];
    }

    delete [] filter;
}

std::string Kernel::getType() {
    return this->type;
}
