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
    Input: the height and width of image
    Must use free_image() to de-allocate
*/
image allocate_image(int width, int height);

/* 
    Description: De-allocate memory of the image and reset all parameters
*/
void free_image(image& img);


/* 
    Description: Load Image from the given path
    Input: file path of image, the height and width of image, image_struct to hold the read image 
    Return: Success or Failure of loading the image
*/
bool load_image(const char* path, int width, int height, image& out_image); 


/* 
    Description: Save Image in the given path
    Input: file path, image to be saved
    Return: Success or Failure of loading the image
*/
bool save_image(const char* path, const image& img);


#endif // IMAGE_IO_H