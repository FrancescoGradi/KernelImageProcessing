//
// Created by federico on 11/03/19.
//

#include "Sharpen.h"

Sharpen::Sharpen(std::string type) : Kernel(type) {
    /*
    for (int i = 0; i < this->size; i++) {
        this->filter[i] = new float[this->size];

        for (int j = 0; j < this->size; j++) {
            if ((i == (this->size / 2)) && (j == (this->size / 2))) {
                this->filter[i][j] = 5;
            }
            else if (((i == this->size/2) || (j == this->size/2)) && (i != j)) {
                this->filter[i][j] = -1;
            }
            else {
                this->filter[i][j] = 0;
            }
        }
    }
    */
    for (int i = 0; i < this->size; i++) {
        for (int j = 0; j < this->size; j++) {
            if ((i == (this->size / 2)) && (j == (this->size / 2))) {
                this->filter[i * size + j] = 5;
            }
            else if (((i == this->size/2) || (j == this->size/2)) && (i != j)) {
                this->filter[i * size + j] = -1;
            }
            else {
                this->filter[i * size + j] = 0;
            }
        }
    }
}
