#include "gaussian.h"
#include <cstdlib>
#include <cstring>

image gaussian_blur(const image& input) {
    // Allocate output image with same dimensions
    image output = allocate_image(input.width, input.height);

    for (int row = 0; row < input.height; row++) {
        for (int col = 0; col < input.width; col++) {
            // Accumulate into int32_t to avoid overflow
            // max value = 255 * 41 * 25 = ~260K fits in int32
            int32_t sum = 0;

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int r = row + ky;
                    int c = col + kx;

                    // Zero-padding: treat out-of-bounds pixels as 0
                    uint8_t pixel = 0;
                    if (r >= 0 && r < input.height &&
                        c >= 0 && c < input.width) {
                        pixel = input.data[r * input.width + c];
                    }

                    sum += (int32_t)pixel * GAUSS_KERNEL[ky + 2][kx + 2];
                }
            }

            // Divide by kernel sum (273) and clamp to [0, 255]
            int32_t result = sum / GAUSS_SUM;
            if (result > 255) result = 255;
            if (result < 0)   result = 0;

            output.data[row * output.width + col] = (uint8_t)result;
        }
    }

    return output;
}
