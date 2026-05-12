#include "hysteresis.h"

/*==============================================================================
                        Hysteresis Edge Tracking
==============================================================================*/
image hysteresis_tracking(
    const image& thresholded
)
{
    image output = allocate_image(
        thresholded.width,
        thresholded.height
    );

    int width  = thresholded.width;
    int height = thresholded.height;

    // Copy thresholded image
    for(int i = 0; i < width * height; i++)
    {
        output.data[i] = thresholded.data[i];
    }

    // Process weak edges
    for(int y = 1; y < height - 1; y++)
    {
        for(int x = 1; x < width - 1; x++)
        {
            int idx = y * width + x;

            // Only check weak edges
            if(output.data[idx] == WEAK_EDGE)
            {
                bool connected = false;

                // Check 8 neighboring pixels
                for(int j = -1; j <= 1; j++)
                {
                    for(int i = -1; i <= 1; i++)
                    {
                        int neighbor_idx =
                            (y + j) * width + (x + i);

                        if(output.data[neighbor_idx]
                            == STRONG_EDGE)
                        {
                            connected = true;
                        }
                    }
                }

                // Keep or remove weak edge
                if(connected)
                {
                    output.data[idx] = STRONG_EDGE;
                }
                else
                {
                    output.data[idx] = NO_EDGE;
                }
            }
        }
    }

    return output;
}