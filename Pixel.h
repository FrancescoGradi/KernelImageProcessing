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
    Pixel(int r, int g, int b):r(r), g(g), b(b) {}
    virtual ~Pixel() {}

    int getR() const {
        return r;
    }

    void setR(int r) {
        Pixel::r = r;
    }

    int getG() const {
        return g;
    }

    void setG(int g) {
        Pixel::g = g;
    }

    int getB() const {
        return b;
    }

    void setB(int b) {
        Pixel::b = b;
    }

private:
    int r;
    int g;
    int b;
};


#endif //KERNELIMAGEPROCESSING_PIXEL_H
