#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "image_io.h"
#include <cstdint>

/*=================================================================================
                            Gaussian Blur Constants
==================================================================================*/
// 5x5 Gaussian kernel (sigma ~1.0)
// Coefficients sum to 273 (as specified in project guide)
// Zero-padding boundary: out-of-bounds pixels treated as 0
static const int GAUSS_KERNEL[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};
static const int GAUSS_SUM = 273;

/*=================================================================================
                            Function Declarations
==================================================================================*/
/*
    Description: Apply 5x5 Gaussian blur to input image
    Input: input image struct
    Output: new blurred image (caller must free with free_image())
    Notes: Uses zero-padding for boundary handling
           Accumulates into int32_t to avoid overflow (max = 255*41*25 = ~260K)
*/
image gaussian_blur(const image& input);

#endif // GAUSSIAN_H
image gaussian_blur_rvv(const image& input);

image gaussian_blur_rvv(const image& input);
image gaussian_blur_rvv(const image& input);
