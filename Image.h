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
    explicit Image(Pixel** pixels, int width, int height, int max, std::string magic);
    explicit Image(std::string pathImage);
    virtual ~Image();

    void loadImage(std::string pathImage);
    void storeImage(std::string pathDest);

    void headerCommentCheck(std::ifstream* picture);


    int getWidth() const;

    void setWidth(int width);

    int getHeight() const;

    void setHeight(int height);

    int getChannels() const;

    void setChannels(int channels);

    std::string getMagic() const;

    void setMagic(std::string magic);

    Pixel **getPixels() const;

    void setPixels(Pixel **pixels);

private:
    int width;
    int height;
    int channels;
    int max;
    std::string magic;
    Pixel** pixels;
};


#endif //KERNELIMAGEPROCESSING_IMAGE_H
