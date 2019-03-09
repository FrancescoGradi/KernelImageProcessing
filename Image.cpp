//
// Created by fra on 06/03/19.
//

#include "Image.h"
#include "Pixel.h"

#include <iostream>
#include <fstream>

Image::Image(): width(0), height(0), channels(0), pixels(nullptr) {}

Image::Image(std::string pathImage) {
    // Costruttore che incapsula il caricamento dell'immagine

    Image::loadImage(pathImage);
}

Image::~Image() {

    if (pixels != nullptr) {

        for(int i = 0; i < width; i++) {
            delete [] pixels[i];
        }

        delete [] pixels;
    }

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

Pixel **Image::getPixels() const {
    return pixels;
}

void Image::setPixels(Pixel **pixels) {
    Image::pixels;
}

void Image::loadImage(const std::string pathImage) {

    std::ifstream picture;
    char* tmp;

    picture.open(pathImage);
    if (picture.fail()) {
        std::cout << "Image loading error." << std::endl;
        return;
    } else {
        std::cout << "Stream ok..." << std::endl;
    }

    // Assegna agli attributi di Image i valori necessari per procedere allo scorrimento del payload
    // Scopre anche le dimensioni dell'immagine e il magic number
    headerCommentCheck(&picture);

    int size = width * height * 3;
    tmp = new char[size];

    // Ho ripreso il codice di loro che prevedeva dei char per leggere i byte
    picture.read(tmp, size);

    pixels = new Pixel*[height];

    std::string byteRead = "";
    for(int i = 0; i < height; i++) {
        pixels[i] = new Pixel[width];

        for(int j = 0; j < width; j++) {

            pixels[i][j].setR((unsigned char) tmp[3*i*width + 3*j + 0]);
            pixels[i][j].setG((unsigned char) tmp[3*i*width + 3*j + 1]);
            pixels[i][j].setB((unsigned char) tmp[3*i*width + 3*j + 2]);

        }
    }

    picture.close();
}

void Image::headerCommentCheck(std::ifstream* picture) {
    /*
     *
     * Ci sono dei problemi qua
     * TODO Fix (la seconda parte non commentata sembra funzionare)
     *
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
                std::cout << "qui";
            }
            else if (i == 1) {
                this->width = atoi(byteToCheck.c_str());
                std::cout << "quo";
            }
            else if (i == 2) {
                this->height = atoi(byteToCheck.c_str());
                std::cout << "qua";

            }
        }
    }
    */
    std::string byteToCheck = "";
    bool isComment = false;

    while (!isComment) {
        *picture >> byteToCheck;
        if (byteToCheck == "#")
            std::getline(*picture, byteToCheck);
        else
            isComment = true;
        magic = byteToCheck;
    }

    isComment = false;

    while (!isComment) {
        *picture >> byteToCheck;
        if (byteToCheck == "#")
            std::getline(*picture, byteToCheck);
        else
            isComment = true;
        //a va convertito in intero
        width = atoi(byteToCheck.c_str());
    }

    isComment = false;

    while (!isComment) {
        *picture >> byteToCheck;
        if (byteToCheck == "#")
            std::getline(*picture, byteToCheck);
        else
            isComment = true;
        height = atoi(byteToCheck.c_str());
    }

}
