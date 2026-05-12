#ifndef CANNY_POST_H
#define CANNY_POST_H

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

image non_maximum_suppression(
    const image& magnitude,
    const image& direction
);

image double_threshold(
    const image& nms_image,
    uint8_t low_threshold,
    uint8_t high_threshold
);

#endif