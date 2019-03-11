#include "Image.h"
#include "Kernel.h"
#include "KernelFactory.h"

#include <iostream>
#include <fstream>


int main() {

    Image* img = new Image("../images/computer_programming.ppm");

    std::cout << img->getWidth() << std::endl;
    std::cout << img->getHeight() << std::endl;
    std::cout << img->getMagic() << std::endl;

    int n = 3;
    int m = 5;
    auto* kf = new KernelFactory();

    Kernel* blur = kf->createKernel(m, "gauss");
    Kernel* identity = kf->createKernel(m, "identity");
    Kernel* sharpen = kf->createKernel(m, "sharpen");
    Kernel* boxBlur = kf->createKernel(m, "box");
    Kernel* edges = kf->createKernel(m, "edges");

    auto newImage = edges->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/edges5.ppm");

    newImage = blur->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/blur5.ppm");

    newImage = identity->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/identity5.ppm");

    newImage = sharpen->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/sharpen5.ppm");

    newImage = boxBlur->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/boxBlur5.ppm");

    Kernel* blur3 = kf->createKernel(n, "gauss");
    Kernel* identity3 = kf->createKernel(n, "identity");
    Kernel* sharpen3 = kf->createKernel(n, "sharpen");
    Kernel* boxBlur3 = kf->createKernel(n, "box");
    Kernel* edges3 = kf->createKernel(n, "edges");

    newImage = edges3->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/edges3.ppm");

    newImage = blur3->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/blur3.ppm");

    newImage = identity3->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/identity3.ppm");

    newImage = sharpen3->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/sharpen3.ppm");

    newImage = boxBlur3->applyFiltering(img->getPixels(), img->getWidth(), img->getHeight(), img->getMagic());
    newImage->storeImage("../images/boxBlur3.ppm");

    delete blur;
    delete identity;
    delete sharpen;
    delete boxBlur;
    delete edges;

    delete blur3;
    delete identity3;
    delete sharpen3;
    delete boxBlur3;
    delete edges3;

    return 0;

}