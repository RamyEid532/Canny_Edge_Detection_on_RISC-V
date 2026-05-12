#include "non-maximum_suppression _and_double_threshold.h"
#include <algorithm>

/*==============================================================================
                    Non-Maximum Suppression
==============================================================================*/
image non_maximum_suppression(
    const image& magnitude,
    const image& direction
)
{
    image output = allocate_image(
        magnitude.width,
        magnitude.height
    );

    int width  = magnitude.width;
    int height = magnitude.height;

    for(int y = 1; y < height - 1; y++)
    {
        for(int x = 1; x < width - 1; x++)
        {
            int idx = y * width + x;

            uint8_t current = magnitude.data[idx];
            uint8_t dir = direction.data[idx];

            uint8_t neighbor1 = 0;
            uint8_t neighbor2 = 0;

            // 0 degrees
            if(dir == 0)
            {
                neighbor1 = magnitude.data[y * width + (x - 1)];
                neighbor2 = magnitude.data[y * width + (x + 1)];
            }

            // 45 degrees
            else if(dir == 45)
            {
                neighbor1 = magnitude.data[(y - 1) * width + (x + 1)];
                neighbor2 = magnitude.data[(y + 1) * width + (x - 1)];
            }

            // 90 degrees
            else if(dir == 90)
            {
                neighbor1 = magnitude.data[(y - 1) * width + x];
                neighbor2 = magnitude.data[(y + 1) * width + x];
            }

            // 135 degrees
            else if(dir == 135)
            {
                neighbor1 = magnitude.data[(y - 1) * width + (x - 1)];
                neighbor2 = magnitude.data[(y + 1) * width + (x + 1)];
            }

            if(current >= neighbor1 &&
               current >= neighbor2)
            {
                output.data[idx] = current;
            }
            else
            {
                output.data[idx] = 0;
            }
        }
    }

    return output;
}

/*==============================================================================
                        Double Thresholding
==============================================================================*/
image double_threshold(
    const image& nms_image,
    uint8_t low_threshold,
    uint8_t high_threshold
)
{
    image output = allocate_image(
        nms_image.width,
        nms_image.height
    );

    int size = nms_image.width * nms_image.height;

    for(int i = 0; i < size; i++)
    {
        uint8_t pixel = nms_image.data[i];

        if(pixel >= high_threshold)
        {
            output.data[i] = STRONG_EDGE;
        }
        else if(pixel >= low_threshold)
        {
            output.data[i] = WEAK_EDGE;
        }
        else
        {
            output.data[i] = NO_EDGE;
        }
    }

    return output;
}
