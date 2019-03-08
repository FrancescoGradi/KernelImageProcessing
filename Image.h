//
// Created by fra on 06/03/19.
//

#ifndef KERNELIMAGEPROCESSING_IMAGE_H
#define KERNELIMAGEPROCESSING_IMAGE_H

#include <string>
#include "Pixel.h"

class Image {

public:
    Image();
    Image(int width, int height, int channels);
    virtual ~Image();

    void loadImage(const std::string pathImage);

    void headerCommentCheck(std::ifstream* picture);

    int getWidth() const;

    void setWidth(int width);

    int getHeight() const;

    void setHeight(int height);

    int getChannels() const;

    void setChannels(int channels);

    std::string getMagic() const;

    void setMagic(std::string magic);

    Pixel *getPixels() const;

    void setPixels(Pixel *pixels, int i, int j);

private:
    int width;
    int height;
    int channels;
    std::string magic;
    Pixel* pixels;
};


#endif //KERNELIMAGEPROCESSING_IMAGE_H
