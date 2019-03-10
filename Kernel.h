//
// Created by fra on 09/03/19.
//

#ifndef KERNELIMAGEPROCESSING_KERNEL_H
#define KERNELIMAGEPROCESSING_KERNEL_H


#include "Image.h"

class Kernel {

public:
    // Dobbiamo trovare un modo semifurbo per gestire le differenti dimensioni delle matrici di kernel.
    // Pensavo di chiamare un metodo statico che ti rendesse direttamente la matrice richiesta, magari
    // avendo come argomento l'n delle dimensioni. Per ora si usano metodi statici che possono essere
    // chiamati anche senza instanziare un oggetto, ma lo svantaggio e' che dobbiamo usare un delete una
    // volta che la matrice non ci serve piu' (penso che in teoria in questi casi il distruttore non
    // intervenga.)

    Kernel(int size, std::string type);
    virtual ~Kernel();

    std::string getType();

    float** getFilter();

    static Image* applyFiltering(Image* img, float** filter); // sarò la funzione che fa il filtraggio

    // Altre matrici da utilizzare: blur, sharpen, sobel, edgeDetection, gaussian blur

protected:
    float** filter;

private:
    int size;
    std::string type;

};


#endif //KERNELIMAGEPROCESSING_KERNEL_H
