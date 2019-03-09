//
// Created by fra on 09/03/19.
//

#include "Kernel.h"
#include "Pixel.h"

// Restituisce il kernel piu' semplice, che non modifica l'immagine. Ha un 1 nel mezzo.
float** Kernel::getIdentity(int n) {

    auto identity = new float*[n];

    for (int i = 0; i < n; i++) {
        identity[i] = new float[n];

        for (int j = 0; j < n; j++) {
            if ((i == (n/2)) && (j == (n/2))) {
                identity[i][j] = 1;
            } else {
                identity[i][j] = 0;
            }
        }
    }

    return identity;
}

float** Kernel::getGaussianBlur() {

    auto blur = new float*[3];

    for (int i = 0; i < 3; i++) {
        blur[i] = new float[3];
    }

    blur[0][0] = float(1)/float(16);
    blur[0][1] = float(1)/float(8);
    blur[0][2] = float(1)/float(16);
    blur[1][0] = float(1)/float(8);
    blur[1][1] = float(1)/float(4);
    blur[1][2] = float(1)/float(8);
    blur[2][0] = float(1)/float(16);
    blur[2][1] = float(1)/float(8);
    blur[2][2] = float(1)/float(16);

    return blur;

}

float** Kernel::getSharpen() {

    auto sharpen = new float*[3];

    for (int i = 0; i < 3; i++) {
        sharpen[i] = new float[3];
    }

    sharpen[0][0] = 0;
    sharpen[0][1] = -1;
    sharpen[0][2] = 0;
    sharpen[1][0] = -1;
    sharpen[1][1] = 5;
    sharpen[1][2] = -1;
    sharpen[2][0] = 0;
    sharpen[2][1] = -1;
    sharpen[2][2] = 0;

    return sharpen;
}