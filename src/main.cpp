#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>

#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "suppression.h"
#include "thresholding.h"
#include "hysteresis.h"

static double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

static void fill_rectangle(image& img) {
    memset(img.data, 0, img.width * img.height);
    for (int y = img.height/4; y < 3*img.height/4; y++)
        for (int x = img.width/4; x < 3*img.width/4; x++)
            img.data[y * img.width + x] = 255;
}

#define WARMUP 20
#define ITERS  200

int main(int argc, char* argv[]) {
    const int width  = 512;
    const int height = 512;

    image image_in = allocate_image(width, height);

    if (argc > 1) {
        if (!load_image(argv[1], width, height, image_in)) {
            printf("Failed to load image!\n");
            return 1;
        }
        printf("Using real image: %s\n", argv[1]);
    } else {
        fill_rectangle(image_in);
        printf("Using generated rectangle image.\n");
    }

    printf("Canny Edge Detection - Per-Stage Timing\n");
    printf("Image: %dx%d | Warmup: %d | Iterations: %d\n", width, height, WARMUP, ITERS);
    printf("==========================================\n");

    // WARMUP
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
    // Stage 2a: Sobel Gx/Gy
    // -------------------------------------------------------
    int16_t* gx_buf = new int16_t[width * height]();
    int16_t* gy_buf = new int16_t[width * height]();

    double t_gxgy = 0;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        compute_gxgy(last_blurred, gx_buf, gy_buf);
        t_gxgy += now_ns() - s;
    }
    t_gxgy /= ITERS;

    // -------------------------------------------------------
    // Stage 2b: Magnitude
    // -------------------------------------------------------
    image mag_l1 = allocate_image(width, height);
    image mag_l2 = allocate_image(width, height);

    double t_mag = 0;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        compute_magnitude(gx_buf, gy_buf, width * height, mag_l1, mag_l2);
        t_mag += now_ns() - s;
    }
    t_mag /= ITERS;

    // -------------------------------------------------------
    // Stage 2c: Direction
    // -------------------------------------------------------
    image direction = allocate_image(width, height);

    double t_dir = 0;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        compute_direction(gx_buf, gy_buf, width * height, direction);
        t_dir += now_ns() - s;
    }
    t_dir /= ITERS;

    // -------------------------------------------------------
    // Stage 3: Non-Maximum Suppression
    // -------------------------------------------------------
    double t_nms = 0;
    image last_nms = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image n = non_maximum_suppression(mag_l1, direction);
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
        image t = double_threshold(last_nms, 3, 15);
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

    // Save output
    image final_output = hysteresis_tracking(last_thresh);
    printf("Saving output: %dx%d\n", final_output.width, final_output.height);
    printf("Data ptr: %p\n", (void*)final_output.data);
    save_image("output.raw", final_output);
    free_image(final_output);

    // -------------------------------------------------------
    // Results
    // -------------------------------------------------------
    double t_total = t_gaussian + t_gxgy + t_mag + t_dir
                   + t_nms + t_thresh + t_hysteresis;

    printf("\nPer-Stage Breakdown:\n");
    printf("------------------------------------------\n");
    printf("Stage 1 - Gaussian Blur:         %7.3f ms  (%4.1f%%)\n",
           t_gaussian/1e6, 100.0*t_gaussian/t_total);
    printf("Stage 2a- Sobel Gx/Gy:           %7.3f ms  (%4.1f%%)\n",
           t_gxgy/1e6, 100.0*t_gxgy/t_total);
    printf("Stage 2b- Magnitude:             %7.3f ms  (%4.1f%%)\n",
           t_mag/1e6, 100.0*t_mag/t_total);
    printf("Stage 2c- Direction:             %7.3f ms  (%4.1f%%)\n",
           t_dir/1e6, 100.0*t_dir/t_total);
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
    delete[] gx_buf;
    delete[] gy_buf;
    free_image(mag_l1);
    free_image(mag_l2);
    free_image(direction);
    free_image(last_nms);
    free_image(last_thresh);

    return 0;
}
