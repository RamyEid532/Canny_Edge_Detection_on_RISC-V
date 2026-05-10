#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>



/*=================================================================================
                                  Type Definitons
==================================================================================*/
struct image{
    int width;
    int height;
    uint8_t* data;
};



/*=================================================================================
                                  Functions Declarations
==================================================================================*/

/* 
    Description: Allocate memory for a blank image
    Input: takes the height and width of image
    Must use free_image() to de-allocate
*/
image allocate_image(int width, int height);

/* 
    Description: De-allocate memory of the image and resets all parameters
*/
void free_image(image& img);


#endif // IMAGE_IO_H