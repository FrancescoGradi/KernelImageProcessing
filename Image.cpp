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

Pixel *Image::getPixels() const {
    return pixels;
}

void Image::setPixels(Pixel *pixels) {
    Image::pixels = pixels;
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

    // Adesso bisogna analizzare il flusso in modo da capire gli header di formato PPM

    picture.close();
}
