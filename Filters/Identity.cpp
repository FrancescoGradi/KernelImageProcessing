//
// Created by federico on 10/03/19.
//

#include "Identity.h"

Identity::Identity(std::string type, int size) : Kernel(size, type) {

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if ((i == (size / 2)) && (j == (size / 2))) {
                this->filter[i*size + j] = 1;
            } else {
                this->filter[i*size + j] = 0;
            }
        }
    }
}
