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
// Helper: get nanoseconds
// =========================================================================
static double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// =========================================================================
// Measure single stage
// =========================================================================
#define WARMUP 20
#define ITERS  200

int main() {
    const int width  = 512;
    const int height = 512;

    // Allocate input image
    image image_in = allocate_image(width, height);
    for (int i = 0; i < width * height; i++)
        image_in.data[i] = (uint8_t)(i % 256);

    printf("Canny Edge Detection - Per-Stage Timing\n");
    printf("Image: %dx%d | Warmup: %d | Iterations: %d\n", width, height, WARMUP, ITERS);
    printf("==========================================\n");

    // -------------------------------------------------------
    // WARMUP: شغّل الـ pipeline كام مرة قبل القياس
    // -------------------------------------------------------
    for (int i = 0; i < WARMUP; i++) {
        image b = gaussian_blur(image_in);
        sobel_result s = sobel_gradient(b);
        image n = non_maximum_suppression(s.magnitude_l1, s.direction);
        image t = double_threshold(n, 50, 150);
        image f = hysteresis_tracking(t);
        free_image(b);
        free_sobel_result(s);
        free_image(n);
        free_image(t);
        free_image(f);
    }

    // -------------------------------------------------------
    // Stage 1: Gaussian Blur
    // -------------------------------------------------------
    double t_gaussian = 0;
    image last_blurred = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image b = gaussian_blur(image_in);
        t_gaussian += now_ns() - s;
        if (i == ITERS - 1) {
            free_image(last_blurred);
            last_blurred = b;
        } else {
            free_image(b);
        }
    }
    t_gaussian /= ITERS;

    // -------------------------------------------------------
    // Stage 2: Sobel Gradient
    // -------------------------------------------------------
    double t_sobel = 0;
    sobel_result last_sobel;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        sobel_result sr = sobel_gradient(last_blurred);
        t_sobel += now_ns() - s;
        if (i == ITERS - 1) {
            last_sobel = sr;
        } else {
            free_sobel_result(sr);
        }
    }
    t_sobel /= ITERS;

    // -------------------------------------------------------
    // Stage 3: Non-Maximum Suppression
    // -------------------------------------------------------
    double t_nms = 0;
    image last_nms = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image n = non_maximum_suppression(last_sobel.magnitude_l1, last_sobel.direction);
        t_nms += now_ns() - s;
        if (i == ITERS - 1) {
            free_image(last_nms);
            last_nms = n;
        } else {
            free_image(n);
        }
    }
    t_nms /= ITERS;

    // -------------------------------------------------------
    // Stage 4: Double Thresholding
    // -------------------------------------------------------
    double t_thresh = 0;
    image last_thresh = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image t = double_threshold(last_nms, 50, 150);
        t_thresh += now_ns() - s;
        if (i == ITERS - 1) {
            free_image(last_thresh);
            last_thresh = t;
        } else {
            free_image(t);
        }
    }
    t_thresh /= ITERS;

    // -------------------------------------------------------
    // Stage 5: Hysteresis Tracking
    // -------------------------------------------------------
    double t_hysteresis = 0;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image f = hysteresis_tracking(last_thresh);
        t_hysteresis += now_ns() - s;
        free_image(f);
    }
    t_hysteresis /= ITERS;

    // -------------------------------------------------------
    // Results
    // -------------------------------------------------------
    double t_total = t_gaussian + t_sobel + t_nms + t_thresh + t_hysteresis;

    printf("\nPer-Stage Breakdown:\n");
    printf("------------------------------------------\n");
    printf("Stage 1 - Gaussian Blur:         %7.3f ms  (%4.1f%%)\n",
           t_gaussian/1e6, 100.0*t_gaussian/t_total);
    printf("Stage 2 - Sobel Gradient:        %7.3f ms  (%4.1f%%)\n",
           t_sobel/1e6, 100.0*t_sobel/t_total);
    printf("Stage 3 - Non-Max Suppression:   %7.3f ms  (%4.1f%%)\n",
           t_nms/1e6, 100.0*t_nms/t_total);
    printf("Stage 4 - Double Threshold:      %7.3f ms  (%4.1f%%)\n",
           t_thresh/1e6, 100.0*t_thresh/t_total);
    printf("Stage 5 - Hysteresis Tracking:   %7.3f ms  (%4.1f%%)\n",
           t_hysteresis/1e6, 100.0*t_hysteresis/t_total);
    printf("------------------------------------------\n");
    printf("Total Pipeline:                  %7.3f ms\n", t_total/1e6);
    printf("==========================================\n");

    // Cleanup
    free_image(image_in);
    free_image(last_blurred);
    free_sobel_result(last_sobel);
    free_image(last_nms);
    free_image(last_thresh);

    return 0;
}
