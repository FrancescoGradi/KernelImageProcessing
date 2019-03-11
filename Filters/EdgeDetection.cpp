//
// Created by federico on 11/03/19.
//

#include "EdgeDetection.h"

// I valori per i filtri li ho trovati sulla pagina di wikipedia linkata nella guida del prof

EdgeDetection::EdgeDetection(std::string type, int size) : Kernel(size, type) {

    for (int i = 0; i < size; i++) {
        this->filter[i] = new float[size];

        for (int j = 0; j < size; j++) {
            if ((i == (size / 2)) && (j == (size / 2))) {
                this->filter[i][j] = 8;
            } else {
                this->filter[i][j] = -1;
            }
        }
    }

}
