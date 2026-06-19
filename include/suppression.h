#ifndef SUPRESSION_H
#define SUPRESSION_H

#include "sobel.h"
#include "image_io.h"
#include <cstdint>

/*==============================================================================
                        Function Declarations
==============================================================================*/

image non_maximum_suppression(
    const image& magnitude,
    const image& direction
);

image non_maximum_suppression_rvv(const image& magnitude, const image& direction);

#endif
