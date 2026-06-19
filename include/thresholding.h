#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include "sobel.h"
#include "image_io.h"
#include <cstdint>

/*==============================================================================
                        Double Threshold Constants
==============================================================================*/
static const uint8_t STRONG_EDGE = 255;
static const uint8_t WEAK_EDGE   = 75;
static const uint8_t NO_EDGE     = 0;

/*==============================================================================
                        Function Declarations
==============================================================================*/
image double_threshold(
    const image& nms_image,
    uint8_t low_threshold,
    uint8_t high_threshold
);

image double_threshold_rvv(const image& nms_image, uint8_t low, uint8_t high);
#endif
