//
// Created by fra on 06/03/19.
//

#ifndef KERNELIMAGEPROCESSING_IMAGE_H
#define KERNELIMAGEPROCESSING_IMAGE_H

#include <string>

class Image {

public:
    Image();
    explicit Image(float* pixels, int width, int height, int max, int channels, std::string magic);
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

    float *getPixels() const;

    void setPixels(float *pixels);

private:
    int width;
    int height;
    int max;
    int channels;
    std::string magic;
    float* pixels;
};


#endif //KERNELIMAGEPROCESSING_IMAGE_H
