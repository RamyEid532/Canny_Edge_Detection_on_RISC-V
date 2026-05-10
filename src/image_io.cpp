#include "image_io.h"
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cstdlib>



image allocate_image(int width, int height)
{

    assert(width > 0 && height > 0);
    
    image img;
    img.width = width;
    img.height = height;


    size_t size = static_cast<size_t>(width) * height;
    // Round up to next 64-byte boundary
    size_t alloc_size = (size + 63) & ~(size_t)63;
    img.data = static_cast<uint8_t*>(aligned_alloc(64, alloc_size));

    // Warning in case of failure in allocation
    if (!img.data) {
        fprintf(stderr, "[image_io] aligned_alloc failed for %dx%d image\n", width, height);
        return img;
    }
    memset(img.data, 0, alloc_size);

    return img;
}


void free_image(image& img) {
    free(img.data); // Free allocated memory
    img.data   = nullptr;
    img.width  = 0;
    img.height = 0;
}


bool load_image(const char* path, int width, int height, image& out_image) 
{
    FILE* f = fopen(path, "rb"); //Open image file path
    // Failed in opening the file 
    if (!f) {
        fprintf(stderr, "[image_io] Cannot open '%s' for reading\n", path);
        return false;
    }

    out_image = allocate_image(width, height);
    // Failed allocation 
    if (!out_image.data) { 
        fclose(f); 
        return false; 
    }

    size_t expected_size = (size_t)width * height;
    // Read the image
    size_t actual_size = fread(out_image.data, 1, expected_size, f);
    fclose(f);

    // Size mismatch
    if (actual_size != expected_size) {
        fprintf(stderr, "[image_io] Expected %zu bytes, got %zu from '%s'\n",
                expected_size, actual_size, path);
        
        free_image(out_image);
        return false;
    }

    return true;
}


