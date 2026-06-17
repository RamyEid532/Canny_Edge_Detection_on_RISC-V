#ifndef SOBEL_H
#define SOBEL_H

#include "image_io.h"
#include <cstdint>

static const int SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

static const int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

struct sobel_result {
    image magnitude_l1;
    image magnitude_l2;
    image direction;
    int16_t* Gx;
    int16_t* Gy;
};

// الـ function الأصلية
sobel_result sobel_gradient(const image& input);

// الـ functions الجديدة المفصولة للـ profiling
void compute_gxgy(const image& input, int16_t* Gx, int16_t* Gy);
void compute_magnitude(int16_t* Gx, int16_t* Gy, int size,
                       image& mag_l1, image& mag_l2);
void compute_direction(int16_t* Gx, int16_t* Gy, int size, image& direction);

void free_sobel_result(sobel_result& result);

#endif
