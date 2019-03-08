#include "Image.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>


int main() {

    Image* img = new Image("../computer_programming.ppm");

    std::cout << img->getWidth() << std::endl;
    std::cout << img->getHeight() << std::endl;
    std::cout << img->getMagic() << std::endl;

    for(int i = 0; i < img->getWidth(); i++) {
        for(int j = 0; j < img->getHeight(); j++) {

            std::cout << img->getPixels()[i][j].getR() << std::endl;
        }
    }

    return 0;
}