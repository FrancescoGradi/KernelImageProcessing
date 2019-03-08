#include "Image.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>


int main() {

    Image* img = new Image(2, 2, 3);
    img->loadImage("../computer_programming.ppm");


    return 0;
}