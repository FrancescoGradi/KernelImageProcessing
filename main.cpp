#include "Image.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>


int main() {

    Image* img = new Image(508, 493, 3);
    img->loadImage("../computer_programming.ppm");

    std::cout << img->getWidth() << std::endl;
    std::cout << img->getHeight() << std::endl;
    std::cout << img->getMagic() << std::endl;
    std::cout << img->getPixels() << std::endl;

    return 0;
}