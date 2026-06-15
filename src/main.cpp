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

static double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

#define WARMUP 20
#define ITERS  200

int main() {
    const int width  = 512;
    const int height = 512;

    // Load real image instead of synthetic
    image image_in = allocate_image(width, height);
    if (!load_image("/input.raw", width, height, image_in)) {
        printf("Warning: could not load input.raw, using synthetic image\n");
        for (int i = 0; i < width * height; i++)
            image_in.data[i] = (uint8_t)(i % 256);
    }

    printf("Canny Edge Detection - Scalar vs RVV Comparison\n");
    printf("Image: %dx%d | Warmup: %d | Iterations: %d\n", width, height, WARMUP, ITERS);
    printf("=================================================\n");

    // WARMUP
    for (int i = 0; i < WARMUP; i++) {
        image b = gaussian_blur(image_in);
        sobel_result s = sobel_gradient(b);
        image n = non_maximum_suppression(s.magnitude_l1, s.direction);
        image t = double_threshold(n, 50, 150);
        image f = hysteresis_tracking(t);
        free_image(b); free_sobel_result(s);
        free_image(n); free_image(t); free_image(f);
    }

    // SCALAR: Gaussian
    double t_gaussian = 0;
    image last_blurred = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image b = gaussian_blur(image_in);
        t_gaussian += now_ns() - s;
        if (i == ITERS - 1) { free_image(last_blurred); last_blurred = b; }
        else free_image(b);
    }
    t_gaussian /= ITERS;

    // SCALAR: Sobel
    double t_sobel = 0;
    sobel_result last_sobel;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        sobel_result sr = sobel_gradient(last_blurred);
        t_sobel += now_ns() - s;
        if (i == ITERS - 1) last_sobel = sr;
        else free_sobel_result(sr);
    }
    t_sobel /= ITERS;

    // RVV: Gaussian
    double t_gaussian_rvv = 0;
    image last_blurred_rvv = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image b = gaussian_blur_rvv(image_in);
        t_gaussian_rvv += now_ns() - s;
        if (i == ITERS - 1) { free_image(last_blurred_rvv); last_blurred_rvv = b; }
        else free_image(b);
    }
    t_gaussian_rvv /= ITERS;

    // RVV: Sobel
    double t_sobel_rvv = 0;
    sobel_result last_sobel_rvv;
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        sobel_result sr = sobel_gradient_rvv(last_blurred_rvv);
        t_sobel_rvv += now_ns() - s;
        if (i == ITERS - 1) last_sobel_rvv = sr;
        else free_sobel_result(sr);
    }
    t_sobel_rvv /= ITERS;

    // SCALAR: remaining stages
    double t_nms = 0, t_thresh = 0, t_hysteresis = 0;
    image last_nms = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image n = non_maximum_suppression(last_sobel.magnitude_l1, last_sobel.direction);
        t_nms += now_ns() - s;
        if (i == ITERS - 1) { free_image(last_nms); last_nms = n; }
        else free_image(n);
    }
    t_nms /= ITERS;

    image last_thresh = allocate_image(width, height);
    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image t = double_threshold(last_nms, 50, 150);
        t_thresh += now_ns() - s;
        if (i == ITERS - 1) { free_image(last_thresh); last_thresh = t; }
        else free_image(t);
    }
    t_thresh /= ITERS;

    for (int i = 0; i < ITERS; i++) {
        double s = now_ns();
        image f = hysteresis_tracking(last_thresh);
        t_hysteresis += now_ns() - s;
        free_image(f);
    }
    t_hysteresis /= ITERS;

    // Save outputs for correctness verification
    save_image("out_gaussian.raw",  last_blurred);
    save_image("out_sobel_mag.raw", last_sobel.magnitude_l1);
    printf("Outputs saved to out_gaussian.raw and out_sobel_mag.raw\n");

    // Correctness check scalar vs RVV
    int mismatch_gauss = 0, mismatch_sobel = 0;
    for (int i = 0; i < width * height; i++) {
        if (last_blurred.data[i] != last_blurred_rvv.data[i]) mismatch_gauss++;
        if (last_sobel.magnitude_l1.data[i] != last_sobel_rvv.magnitude_l1.data[i]) mismatch_sobel++;
    }

    // Print Results
    double t_total_scalar = t_gaussian + t_sobel + t_nms + t_thresh + t_hysteresis;
    double t_total_rvv    = t_gaussian_rvv + t_sobel_rvv + t_nms + t_thresh + t_hysteresis;

    printf("\n--- SCALAR Results ---\n");
    printf("Gaussian Blur:        %7.3f ms\n", t_gaussian/1e6);
    printf("Sobel Gradient:       %7.3f ms\n", t_sobel/1e6);
    printf("Non-Max Suppression:  %7.3f ms\n", t_nms/1e6);
    printf("Double Threshold:     %7.3f ms\n", t_thresh/1e6);
    printf("Hysteresis Tracking:  %7.3f ms\n", t_hysteresis/1e6);
    printf("Total:                %7.3f ms\n", t_total_scalar/1e6);

    printf("\n--- RVV Results ---\n");
    printf("Gaussian Blur RVV:    %7.3f ms  (speedup: %.2fx)\n",
           t_gaussian_rvv/1e6, t_gaussian/t_gaussian_rvv);
    printf("Sobel Gradient RVV:   %7.3f ms  (speedup: %.2fx)\n",
           t_sobel_rvv/1e6, t_sobel/t_sobel_rvv);
    printf("Total RVV pipeline:   %7.3f ms  (speedup: %.2fx)\n",
           t_total_rvv/1e6, t_total_scalar/t_total_rvv);

    printf("\n--- Correctness Check (Scalar vs RVV) ---\n");
    printf("Gaussian mismatches:  %d / %d %s\n",
           mismatch_gauss, width*height, mismatch_gauss==0 ? "PASS" : "FAIL");
    printf("Sobel mismatches:     %d / %d %s\n",
           mismatch_sobel, width*height, mismatch_sobel==0 ? "PASS" : "FAIL");

    free_image(image_in);
    free_image(last_blurred);
    free_image(last_blurred_rvv);
    free_sobel_result(last_sobel);
    free_sobel_result(last_sobel_rvv);
    free_image(last_nms);
    free_image(last_thresh);

    return 0;
}
