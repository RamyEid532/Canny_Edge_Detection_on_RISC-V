#include "sobel.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

// -------------------------------------------------------
// Stage 2a: Gx و Gy 
// -------------------------------------------------------
void compute_gxgy(const image& input, int16_t* Gx, int16_t* Gy) {
    int W = input.width;
    int H = input.height;

    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int r = row + ky;
                    int c = col + kx;
                    uint8_t pixel = 0;
                    if (r >= 0 && r < H && c >= 0 && c < W)
                        pixel = input.data[r * W + c];
                    gx += pixel * SOBEL_X[ky+1][kx+1];
                    gy += pixel * SOBEL_Y[ky+1][kx+1];
                }
            }
            Gx[row * W + col] = (int16_t)gx;
            Gy[row * W + col] = (int16_t)gy;
        }
    }
}

// -------------------------------------------------------
// Stage 2b: Magnitude 
// -------------------------------------------------------
void compute_magnitude(int16_t* Gx, int16_t* Gy, int size,
                       image& mag_l1, image& mag_l2) {
    int max_l1 = 1;
    float max_l2 = 1.0f;

    // Pass 1: find max
    for (int i = 0; i < size; i++) {
        int l1 = std::abs(Gx[i]) + std::abs(Gy[i]);
        max_l1 = std::max(max_l1, l1);
        float l2 = std::sqrt((float)Gx[i]*Gx[i] + (float)Gy[i]*Gy[i]);
        max_l2 = std::max(max_l2, l2);
    }

    // Pass 2: normalize
    for (int i = 0; i < size; i++) {
        int l1 = std::abs(Gx[i]) + std::abs(Gy[i]);
        mag_l1.data[i] = (uint8_t)(l1 * 255 / max_l1);

        float l2 = std::sqrt((float)Gx[i]*Gx[i] + (float)Gy[i]*Gy[i]);
        mag_l2.data[i] = (uint8_t)(l2 * 255.0f / max_l2);
    }
}

// -------------------------------------------------------
// Stage 2c: Direction
// -------------------------------------------------------
void compute_direction(int16_t* Gx, int16_t* Gy, int size, image& direction) {
    for (int i = 0; i < size; i++) {
        int ax = std::abs(Gx[i]);
        int ay = std::abs(Gy[i]);
        uint8_t dir;
        if      (ay * 5 < ax * 2)  dir = 0;
        else if (ay * 5 < ax * 12) dir = 45;
        else if (ax * 5 < ay * 2)  dir = 90;
        else                       dir = 135;
        direction.data[i] = dir;
    }
}

// -------------------------------------------------------
// Complete function
// -------------------------------------------------------
sobel_result sobel_gradient(const image& input) {
    int W = input.width;
    int H = input.height;

    sobel_result result;
    result.magnitude_l1 = allocate_image(W, H);
    result.magnitude_l2 = allocate_image(W, H);
    result.direction    = allocate_image(W, H);
    result.Gx = new int16_t[W * H]();
    result.Gy = new int16_t[W * H]();

    compute_gxgy(input, result.Gx, result.Gy);
    compute_magnitude(result.Gx, result.Gy, W*H,
                      result.magnitude_l1, result.magnitude_l2);
    compute_direction(result.Gx, result.Gy, W*H, result.direction);

    return result;
}

void free_sobel_result(sobel_result& result) {
    free_image(result.magnitude_l1);
    free_image(result.magnitude_l2);
    free_image(result.direction);
    delete[] result.Gx;
    delete[] result.Gy;
    result.Gx = nullptr;
    result.Gy = nullptr;
}
