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

    int n = 3;
    int m = 5;
    auto* kf = new KernelFactory();

    Kernel* blur = kf->createKernel(n, "gauss");
    std::cout << "Blur:" << std::endl;
    blur->getFilter();

    Kernel* identity = kf->createKernel(m, "identity");
    std::cout << "Identity:" << std::endl;
    identity->getFilter();

    Kernel* sharpen = kf->createKernel(m, "sharpen");
    std::cout << "Sharpen:" << std::endl;
    sharpen->getFilter();

    Kernel* boxBlur = kf->createKernel(m, "box");
    std::cout << "Box Blur:" << std::endl;
    boxBlur->getFilter();

    Kernel* edges = kf->createKernel(m, "edges");
    std::cout << "Edges detection:" << std::endl;
    edges->getFilter();

    auto newImage = blur->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->loadImage("../prova.ppm");

    delete blur;
    delete identity;
    delete sharpen;
    delete boxBlur;
    delete edges;

    return 0;

}