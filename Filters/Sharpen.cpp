//
// Created by federico on 11/03/19.
//

#include "Sharpen.h"

Sharpen::Sharpen(std::string type, int size) : Kernel(size, type) {

    for (int i = 0; i < size; i++) {
        this->filter[i] = new float[size];

        for (int j = 0; j < size; j++) {
            if ((i == (size / 2)) && (j == (size / 2))) {
                this->filter[i][j] = 5;
            }
            else if (((i == size/2) || (j == size/2)) && (i != j)) {
                this->filter[i][j] = -1;
            }
            else {
                this->filter[i][j] = 0;
            }
        }
    }

}
