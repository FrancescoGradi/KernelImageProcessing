//
// Created by federico on 10/03/19.
//

#include "GaussianBlur.h"
#include <math.h>

GaussianBlur::GaussianBlur(std::string type, int size) : Kernel(size, type) {
    /*
    // Double perché l'exp ritorna un double, dava problemi
    double sigma = 1;
    double mean = size/2;
    double sum = 0.0;

    for (int x = 0; x < size; ++x) {
        this->filter[x] = new float[size];
        for (int y = 0; y < size; ++y) {
            this->filter[x][y] = static_cast<float>(
                    exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
                    / (2 * M_PI * sigma * sigma));

            // Accumula i valori
            sum += this->filter[x][y];
        }
    }

    // Normalizza sulla somma
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            this->filter[x][y] /= sum;
        }
    }
    */
    // Double perché l'exp ritorna un double, dava problemi
    double sigma = 1;
    double mean = size/2;
    double sum = 0.0;

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            this->filter[x * size + y] = static_cast<float>(
                    exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
                    / (2 * M_PI * sigma * sigma));

            // Accumula i valori
            sum += this->filter[x * size + y];
        }
    }

    // Normalizza sulla somma
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            this->filter[x * size + y] /= sum;
        }
    }
}
