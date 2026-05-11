#include "sobel.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

sobel_result sobel_gradient(const image& input) {
    int W = input.width;
    int H = input.height;

    // Allocate output images
    sobel_result result;
    result.magnitude_l1 = allocate_image(W, H);
    result.magnitude_l2 = allocate_image(W, H);
    result.direction    = allocate_image(W, H);

    // Structure of Arrays (SoA): better for SIMD/RVV vectorization
    // int16_t sufficient: max = 255 * 4 * 2 = 2040
    result.Gx = new int16_t[W * H]();
    result.Gy = new int16_t[W * H]();

    // Compute Gx and Gy for each pixel
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int gx = 0, gy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int r = row + ky;
                    int c = col + kx;

                    // Zero-padding boundary
                    uint8_t pixel = 0;
                    if (r >= 0 && r < H && c >= 0 && c < W)
                        pixel = input.data[r * W + c];

                    gx += pixel * SOBEL_X[ky+1][kx+1];
                    gy += pixel * SOBEL_Y[ky+1][kx+1];
                }
            }

            result.Gx[row * W + col] = (int16_t)gx;
            result.Gy[row * W + col] = (int16_t)gy;
        }
    }

    // Two-pass normalization: find max then normalize
    int   max_l1 = 1;
    float max_l2 = 1.0f;
    for (int i = 0; i < W * H; i++) {
        int mag_l1 = std::abs(result.Gx[i]) + std::abs(result.Gy[i]);
        max_l1 = std::max(max_l1, mag_l1);

        float mag_l2 = std::sqrt((float)result.Gx[i] * result.Gx[i] +
                                  (float)result.Gy[i] * result.Gy[i]);
        max_l2 = std::max(max_l2, mag_l2);
    }

    // Second pass: compute normalized magnitude and direction
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int i  = row * W + col;
            int gx = result.Gx[i];
            int gy = result.Gy[i];

            // L1 magnitude normalized to [0,255]
            int mag_l1 = std::abs(gx) + std::abs(gy);
            result.magnitude_l1.data[i] = (uint8_t)(mag_l1 * 255 / max_l1);

            // L2 magnitude normalized to [0,255]
            float mag_l2 = std::sqrt((float)gx*gx + (float)gy*gy);
            result.magnitude_l2.data[i] = (uint8_t)(mag_l2 * 255.0f / max_l2);

            // Direction quantization using integer cross-multiplication
            // No atan2() needed: tan(22.5) ~ 2/5, tan(67.5) ~ 12/5
            int ax = std::abs(gx);
            int ay = std::abs(gy);
            uint8_t dir;
            if      (ay * 5 < ax * 2)  dir = 0;
            else if (ay * 5 < ax * 12) dir = 45;
            else if (ax * 5 < ay * 2)  dir = 90;
            else                       dir = 135;
            result.direction.data[i] = dir;
        }
    }

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
