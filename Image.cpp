//
// Created by fra on 06/03/19.
//

#include "Image.h"

#include <iostream>
#include <fstream>

Image::Image(): width(0), height(0), channels(0), pixels(nullptr) {}

Image::Image(int width, int height, int channels) {

    this->width = width;
    this->height = height;
    this->channels = channels;

    Pixel* pixels = nullptr;
}

Image::~Image() {
    delete pixels;
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

int Image::getChannels() const {
    return channels;
}

void Image::setChannels(int channels) {
    Image::channels = channels;
}

std::string Image::getMagic() const {
    return magic;
}

void Image::setMagic(std::string magic) {
    Image::magic = magic;
}

Pixel *Image::getPixels() const {
    return pixels;
}

void Image::setPixels(Pixel *pixels) {
    Image::pixels;
}

void Image::loadImage(const std::string pathImage) {

    std::ifstream picture;
    char* temp;

    picture.open(pathImage);
    if (picture.fail()) {
        std::cout << "Image loading error." << std::endl;
        return;
    } else {
        std::cout << "Stream ok..." << std::endl;
    }

    // Assegna agli attributi di Image i valori necessari per procedere allo scorrimento del payload
    headerCommentCheck(&picture);

    std::string byteRead = "";
    Pixel* pix = new Pixel();
    for(int i = 0; i < this->width; i++) {
        for(int j = 0; j < this->height; j++) {
            Pixel* p = new Pixel();
            picture >> byteRead;
            p->r = atoi(byteRead.c_str());
            picture >> byteRead;
            p->g = atoi(byteRead.c_str());
            picture >> byteRead;
            p->b = atoi(byteRead.c_str());
            this->setPixels(p);
        }
    }

    picture.close();
}

void Image::headerCommentCheck(std::ifstream* picture) {

    std::string byteToCheck = "";
    bool isComment = false;

    for(int i = 0; i < 4; i++) {
        while (!isComment) {
            *picture >> byteToCheck;
            if (byteToCheck == "#")
                std::getline(*picture, byteToCheck);
            else
                isComment = true;
            if(i == 0) {
                this->magic = byteToCheck;
            }
            else if (i == 1) {
                this->width = atoi(byteToCheck.c_str());
            }
            else if (i == 2) {
                this->height = atoi(byteToCheck.c_str());
            }
        }
    }


}
