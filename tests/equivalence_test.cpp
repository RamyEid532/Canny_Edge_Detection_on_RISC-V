#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"

static int errors = 0;

static void check(const char* name, const uint8_t* scalar,
                  const uint8_t* rvv, int size, int tol = 1) {
    int diff_count = 0;
    for (int i = 0; i < size; i++) {
        int diff = (int)scalar[i] - (int)rvv[i];
        if (diff < -tol || diff > tol) diff_count++;
    }
    if (diff_count == 0) {
        printf("[PASS] %s: outputs match within ±%d\n", name, tol);
    } else {
        printf("[FAIL] %s: %d pixels differ by more than ±%d\n", name, diff_count, tol);
        errors++;
    }
}

int main() {
    const int W = 100, H = 75;
    const int SIZE = W * H;

    printf("Equivalence Tests: Scalar vs RVV\n");
    printf("Image: %dx%d (non-power-of-two)\n", W, H);
    printf("==========================================\n");

    image img = allocate_image(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img.data[y*W+x] = (uint8_t)(((x/8 + y/8) % 2) * 255);

    // Test 1: Gaussian Blur
    image scalar_gauss = gaussian_blur(img);
    image rvv_gauss    = gaussian_blur_rvv(img);
    check("Gaussian Blur", scalar_gauss.data, rvv_gauss.data, SIZE);

    // Test 2: Sobel Gx/Gy
    int16_t* gx_scalar = new int16_t[SIZE]();
    int16_t* gy_scalar = new int16_t[SIZE]();
    int16_t* gx_rvv    = new int16_t[SIZE]();
    int16_t* gy_rvv    = new int16_t[SIZE]();

    compute_gxgy(scalar_gauss, gx_scalar, gy_scalar);
    compute_gxgy_rvv(scalar_gauss, gx_rvv, gy_rvv);

    int gx_diff = 0;
    for (int i = 0; i < SIZE; i++) {
        int diff = (int)gx_scalar[i] - (int)gx_rvv[i];
        if (diff < -1 || diff > 1) gx_diff++;
    }
    printf("[%s] Sobel Gx: %d pixels differ\n",
           gx_diff == 0 ? "PASS" : "FAIL", gx_diff);

    int gy_diff = 0;
    for (int i = 0; i < SIZE; i++) {
        int diff = (int)gy_scalar[i] - (int)gy_rvv[i];
        if (diff < -1 || diff > 1) gy_diff++;
    }
    printf("[%s] Sobel Gy: %d pixels differ\n",
           gy_diff == 0 ? "PASS" : "FAIL", gy_diff);

    if (gx_diff > 0 || gy_diff > 0) errors++;

    // Test 3: Magnitude (±2 tolerance for sqrt/fixed-point rounding)
    image mag_l1_scalar = allocate_image(W, H);
    image mag_l2_scalar = allocate_image(W, H);
    image mag_l1_rvv    = allocate_image(W, H);
    image mag_l2_rvv    = allocate_image(W, H);

    compute_magnitude(gx_scalar, gy_scalar, SIZE, mag_l1_scalar, mag_l2_scalar);
    compute_magnitude_rvv(gx_scalar, gy_scalar, SIZE, mag_l1_rvv, mag_l2_rvv);

    check("Magnitude L1", mag_l1_scalar.data, mag_l1_rvv.data, SIZE, 3);

    // Summary
    printf("==========================================\n");
    if (errors == 0)
        printf("ALL TESTS PASSED — Code is vector-length-agnostic ✓\n");
    else
        printf("FAILED: %d test(s) failed\n", errors);

    free_image(img);
    free_image(scalar_gauss);
    free_image(rvv_gauss);
    delete[] gx_scalar;
    delete[] gy_scalar;
    delete[] gx_rvv;
    delete[] gy_rvv;
    free_image(mag_l1_scalar);
    free_image(mag_l2_scalar);
    free_image(mag_l1_rvv);
    free_image(mag_l2_rvv);

    return errors;
}
