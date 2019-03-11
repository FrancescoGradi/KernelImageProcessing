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
    explicit Pixel(char r, char g, char b):r(r), g(g), b(b) {}
    virtual ~Pixel() = default;

    char getR() const {
        return r;
    }

    void setR(char r) {
        Pixel::r = r;
    }

    char getG() const {
        return g;
    }

    void setG(char g) {
        Pixel::g = g;
    }

    char getB() const {
        return b;
    }

    void setB(char b) {
        Pixel::b = b;
    }

private:
    char r;
    char g;
    char b;
};


#endif //KERNELIMAGEPROCESSING_PIXEL_H
