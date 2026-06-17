#ifndef SOBEL_H
#define SOBEL_H

#include "image_io.h"
#include <cstdint>

/*=================================================================================
                            Sobel Gradient Constants
==================================================================================*/
// Gx detects vertical edges (horizontal gradient)
// Gy detects horizontal edges (vertical gradient)
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

/*=================================================================================
                            Type Definitions
==================================================================================*/
/*
    Description: Holds the result of Sobel gradient computation
    - magnitude_l1: L1 norm |Gx|+|Gy| normalized to [0,255]
    - magnitude_l2: L2 norm sqrt(Gx^2+Gy^2) normalized to [0,255]
    - direction: quantized to 0, 45, 90, 135 degrees
    - Gx, Gy: raw gradient arrays in Structure of Arrays (SoA) layout
    Note: SoA layout is better for SIMD/RVV vectorization
          int16_t sufficient: max Sobel output = 255*4*2 = 2040
*/
struct sobel_result {
    image magnitude_l1;
    image magnitude_l2;
    image direction;
    int16_t* Gx;
    int16_t* Gy;
};

/*=================================================================================
                            Function Declarations
==================================================================================*/
/*
    Description: Compute Sobel gradient of input image
    Input: blurred input image
    Output: sobel_result with magnitude (L1+L2) and direction
    Notes: Uses integer cross-multiplication for direction (no atan2 needed)
           Caller must free with free_sobel_result()
*/
sobel_result sobel_gradient(const image& input);

/*
    Description: Free all memory in a sobel_result
*/
void free_sobel_result(sobel_result& result);

#endif // SOBEL_H
sobel_result sobel_gradient_rvv(const image& input);

sobel_result sobel_gradient_rvv(const image& input);
sobel_result sobel_gradient_rvv(const image& input);
