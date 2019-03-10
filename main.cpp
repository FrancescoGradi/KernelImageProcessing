#include "Image.h"
#include "Kernel.h"

#include <iostream>
#include <fstream>


int main() {

    Image* img = new Image("../computer_programming.ppm");

    img->storeImage("../prova.ppm");

    std::cout << img->getWidth() << std::endl;
    std::cout << img->getHeight() << std::endl;
    std::cout << img->getMagic() << std::endl;

    for(int i = 0; i < img->getHeight(); i++) {
        for(int j = 0; j < img->getWidth(); j++) {

            // All'inizio mi dava valori strani a volte, PENSO di aver risolto mettendo al posto dell'int un unsigned
            // char, occupa 1 byte e va da 0 a 255... Castandolo a unsigned int si possono vedere i valori.
            std::cout << " " <<std::endl;
            std::cout << img->getPixels()[i][j].getR() << std::endl;
            std::cout << img->getPixels()[i][j].getG() << std::endl;
            std::cout << img->getPixels()[i][j].getB() << std::endl;

        }
    }

    int n = 3;
    auto identity = Kernel::getIdentity(n);

    std::cout << "Kernel: " << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << identity[i][j] << " ";
        }
        std::cout << " " << std::endl;
    }

    auto blur = Kernel::getGaussianBlur();

    std::cout << "Blur: " << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << blur[i][j] << " ";
        }
        std::cout << " " << std::endl;
    }

    auto sharpen = Kernel::getSharpen();

    std::cout << "Sharpen: " << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << sharpen[i][j] << " ";
        }
        std::cout << " " << std::endl;
    }

    // Evitare memory leak
    for (int i = 0; i < n; i++) {
        delete [] identity[i];
    }

    delete [] identity;

    for (int i = 0; i < 3; i++) {
        delete [] blur[i];
    }

    delete [] blur;

    for (int i = 0; i < 3; i++) {
        delete [] sharpen[i];
    }

    delete [] sharpen;

    return 0;
}