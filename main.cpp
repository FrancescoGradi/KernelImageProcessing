#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>


int main() {

    Image* img = new Image("../computer_programming.ppm");

    img->storeImage("../prova.ppm");

    std::cout << img->getWidth() << std::endl;
    std::cout << img->getHeight() << std::endl;
    std::cout << img->getMagic() << std::endl;

    /*for(int i = 0; i < img->getHeight(); i++) {
        for(int j = 0; j < img->getWidth(); j++) {

            // All'inizio mi dava valori strani a volte, PENSO di aver risolto mettendo al posto dell'int un unsigned
            // char, occupa 1 byte e va da 0 a 255... Castandolo a unsigned int si possono vedere i valori.
            std::cout << " " <<std::endl;
            std::cout << img->getPixels()[i][j].getR() << std::endl;
            std::cout << img->getPixels()[i][j].getG() << std::endl;
            std::cout << img->getPixels()[i][j].getB() << std::endl;

        }
    }*/

    int n = 3;
    int m = 5;
    int p = 5;
    auto* kf = new KernelFactory();

    Kernel* blur = kf->createKernel(n, "gauss");
    std::cout << "Blur:" << std::endl;
    blur->getFilter();

    Kernel* identity = kf->createKernel(m, "identity");
    std::cout << "Identity:" << std::endl;
    identity->getFilter();

    Kernel* sharpen = kf->createKernel(p, "sharpen");
    std::cout << "Sharpen:" << std::endl;
    sharpen->getFilter();

    delete blur;
    delete identity;
    delete sharpen;

    return 0;

}