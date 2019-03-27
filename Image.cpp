//
// Created by fra on 06/03/19.
//

#include "Image.h"

#include <iostream>
#include <fstream>

Image::Image(): width(0), height(0), max(0), pixels(nullptr) {}

Image::Image(float *pixels, int width, int height, int max, int channels, std::string magic) : pixels(pixels),
    width(width), height(height), max(max), channels(channels), magic(magic){}

Image::Image(std::string pathImage) {
    // Costruttore che incapsula il caricamento dell'immagine
    Image::loadImage(pathImage);
}

Image::~Image() {

    if (pixels != nullptr)
        delete [] pixels;
}

int Image::getWidth() const {
    return width;
}

void Image::setWidth(int width) {
    Image::width = width;
}

int Image::getHeight() const {
    return height;
}

void Image::setHeight(int height) {
    Image::height = height;
}

std::string Image::getMagic() const {
    return magic;
}

void Image::setMagic(std::string magic) {
    Image::magic = magic;
}

float *Image::getPixels() const {
    return pixels;
}

void Image::setPixels(float *pixels) {
    Image::pixels;
}

void Image::loadImage(std::string pathImage) {

    std::ifstream picture;
    char* tmp;

    picture.open(pathImage);
    if (picture.fail()) {
        std::cout << "Image loading error." << std::endl;
        return;
    }

    // Assegna agli attributi di Image i valori necessari per procedere allo scorrimento del payload
    // Scopre anche le dimensioni dell'immagine e il magic number
    headerCommentCheck(&picture);

    int size = width * height * 3;
    tmp = new char[size];

    picture.read(tmp, size);

    channels = 3;

    pixels = new float[height * width * channels];

    std::string byteRead = "";
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            pixels[i*width*channels + j*channels + 0] = ((float)(unsigned char) tmp[3*i*width + 3*j + 0]) / 255;
            pixels[i*width*channels + j*channels + 1] = ((float)(unsigned char) tmp[3*i*width + 3*j + 1]) / 255;
            pixels[i*width*channels + j*channels + 2] = ((float)(unsigned char) tmp[3*i*width + 3*j + 2]) / 255;
        }
    }

    picture.close();
}

void Image::headerCommentCheck(std::ifstream* picture) {

    std::string byteToCheck = "";
    bool isComment = false;

    for(int i = 0; i < 4; i++) {
        *picture >> byteToCheck;
        if (byteToCheck == "#")
            std::getline(*picture, byteToCheck);
        else {
            if (i == 0) {
                this->magic = byteToCheck;
            }
            else if (i == 1) {
                this->width = atoi(byteToCheck.c_str());
            }
            else if (i == 2) {
                this->height = atoi(byteToCheck.c_str());
            }
            else if (i == 3) {
                this->max = atoi(byteToCheck.c_str());
            }
        }
    }

}

void Image::storeImage(std::string pathDest) {

    if (pixels == nullptr) {
        std::cout << "No image to store." << std::endl;
        return;
    }

    std::ofstream img;
    img.open(pathDest);

    char* tmp;
    tmp = new char[width * height * 3];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            tmp[3*i*width + 3*j + 2] = (char)(unsigned char)(pixels[i*width*channels + j*channels + 0] * 255);
            tmp[3*i*width + 3*j + 0] = (char)(unsigned char)(pixels[i*width*channels + j*channels + 1] * 255);
            tmp[3*i*width + 3*j + 1] = (char)(unsigned char)(pixels[i*width*channels + j*channels + 2] * 255);
        }
    }

    img << magic << std::endl << width << " " << height << std::endl << std::to_string(max) << std::endl;

    img.write(tmp, width * height * 3);
    img.close();

}

int Image::getChannels() const {
    return channels;
}

void Image::setChannels(int channels) {
    Image::channels = channels;
}
