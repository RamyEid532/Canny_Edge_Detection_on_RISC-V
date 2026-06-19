#ifndef HYSTERESIS_H
#define HYSTERESIS_H

#include "image_io.h"
#include "suppression.h"
#include "thresholding.h"
#include <cstdint>

/*==============================================================================
                    Hysteresis Edge Tracking
==============================================================================*/
/*
    Description:
    Connect weak edges to strong edges

    Input:
    thresholded image

    Output:
    final edge image

    Notes:
    Weak edges connected to strong edges survive.
    Isolated weak edges are removed.
*/
image hysteresis_tracking(
    const image& thresholded
);

image hysteresis_tracking_rvv(const image& thresholded);
#endif // HYSTERESIS_H
