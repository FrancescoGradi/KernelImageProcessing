//
// Created by fra on 06/03/19.
//

#ifndef KERNELIMAGEPROCESSING_PIXEL_H
#define KERNELIMAGEPROCESSING_PIXEL_H

class Pixel {
public:
    Pixel() {
        r = 0;
        g = 0;
        b = 0;
    }
    Pixel(unsigned char r, unsigned char g, unsigned char b):r(r), g(g), b(b) {}
    virtual ~Pixel() = default;

    unsigned char getR() const {
        return r;
    }

    void setR(unsigned char r) {
        Pixel::r = r;
    }

    unsigned char getG() const {
        return g;
    }

    void setG(unsigned char g) {
        Pixel::g = g;
    }

    unsigned char getB() const {
        return b;
    }

    void setB(unsigned char b) {
        Pixel::b = b;
    }

private:
    unsigned char r;
    unsigned char g;
    unsigned char b;
};


#endif //KERNELIMAGEPROCESSING_PIXEL_H
