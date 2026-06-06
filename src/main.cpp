#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <chrono>

#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "suppression.h"
#include "thresholding.h"
#include "hysteresis.h"

// =========================================================================
// Timing Harness (100 iterations average)
// =========================================================================
double measure_canny_pipeline(const image& src, int width, int height) {
    const int ITERS = 100;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < ITERS; i++) {
        // Stage 1: Gaussian Blur
        image blurred = gaussian_blur(src);

        // Stage 2: Sobel Gradient
        sobel_result sobel = sobel_gradient(blurred);

        // Stage 3: Non-Maximum Suppression
        image nms = non_maximum_suppression(sobel.magnitude_l1, sobel.direction);

        // Stage 4: Double Thresholding
        image thresh = double_threshold(nms, 50, 150);

        // Stage 5: Hysteresis Tracking
        image final_edges = hysteresis_tracking(thresh);

        // Free all intermediate images
        free_image(blurred);
        free_sobel_result(sobel);
        free_image(nms);
        free_image(thresh);
        free_image(final_edges);
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return (double)elapsed_ns / ITERS;
}

int main() {
    const int width  = 512;
    const int height = 512;

    // Allocate input image using the project's allocate_image
    image image_in = allocate_image(width, height);

    // Fill with synthetic test pattern
    for (int i = 0; i < width * height; i++) {
        image_in.data[i] = (uint8_t)(i % 256);
    }

    printf("Starting Canny Edge Detection Performance Sweep...\n");
    printf("Image size: %d x %d\n", width, height);
    printf("------------------------------------------\n");

    double avg_ns = measure_canny_pipeline(image_in, width, height);

    printf("Performance Metric: %.3f ms per frame\n", avg_ns / 1e6);
    printf("------------------------------------------\n");

    free_image(image_in);
    return 0;
}
