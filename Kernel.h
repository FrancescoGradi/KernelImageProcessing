//
// Created by fra on 09/03/19.
//

#ifndef KERNELIMAGEPROCESSING_KERNEL_H
#define KERNELIMAGEPROCESSING_KERNEL_H


class Kernel {
public:
    // Dobbiamo trovare un modo semifurbo per gestire le differenti dimensioni delle matrici di kernel.
    // Pensavo di chiamare un metodo statico che ti rendesse direttamente la matrice richiesta, magari
    // avendo come argomento l'n delle dimensioni. Per ora si usano metodi statici che possono essere
    // chiamati anche senza instanziare un oggetto, ma lo svantaggio e' che dobbiamo usare un delete una
    // volta che la matrice non ci serve piu' (penso che in teoria in questi casi il distruttore non
    // intervenga.

    Kernel();
    virtual ~Kernel();

    static float** getIdentity(int n);


    // Altre matrici da utilizzare: blur, sharpen, sobel, edgeDetection, gaussian blur

};


#endif //KERNELIMAGEPROCESSING_KERNEL_H
