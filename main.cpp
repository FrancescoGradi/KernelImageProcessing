#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>


int main() {

    //std::ifstream picture("image.png");
    //std::string path = "image.png";

    //picture.open(path, std::ifstream::in);

    //if (picture.fail()) {
    //    std::cout << "Errore";
    //}

    std::string path = "image.png";

    FILE* file = fopen("/home/fra/Scrivania/KernelImageProcessing/image.png", "rb");

    std::cout << file << std::endl;


    return 0;
}