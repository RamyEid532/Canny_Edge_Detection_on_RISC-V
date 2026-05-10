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
