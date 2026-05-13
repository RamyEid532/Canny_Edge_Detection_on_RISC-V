#include"thresholding.h"

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