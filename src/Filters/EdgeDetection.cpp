//
// Created by federico on 11/03/19.
//

#include "EdgeDetection.h"

// I valori per i filtri li ho trovati sulla pagina di wikipedia linkata nella guida del prof

EdgeDetection::EdgeDetection(std::string type) : Kernel(type) {
    /*
    for (int i = 0; i < this->size; i++) {
        this->filter[i] = new float[this->size];

        for (int j = 0; j < this->size; j++) {
            if ((i == (this->size / 2)) && (j == (this->size / 2))) {
                this->filter[i][j] = 8;
            } else {
                this->filter[i][j] = -1;
            }
        }
    }
    */
    for (int i = 0; i < this->size; i++) {
        for (int j = 0; j < this->size; j++) {
            if ((i == (this->size / 2)) && (j == (this->size / 2))) {
                this->filter[i * size + j] = 8;
            } else {
                this->filter[i * size + j] = -1;
            }
        }
    }
}
