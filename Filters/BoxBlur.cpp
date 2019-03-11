//
// Created by federico on 11/03/19.
//

#include "BoxBlur.h"

BoxBlur::BoxBlur(std::string type, int size) : Kernel(size, type){

    for (int i = 0; i < size; i++) {
        this->filter[i] = new float[size];
        for (int j = 0; j < size; j++) {
            this->filter[i][j] = 1.0 / 9.0;
        }
    }

}
