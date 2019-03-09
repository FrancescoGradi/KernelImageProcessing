//
// Created by fra on 09/03/19.
//

#include "Kernel.h"

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