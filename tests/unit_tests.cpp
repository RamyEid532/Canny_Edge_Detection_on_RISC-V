#include <gtest/gtest.h>
#include <cmath>
#include "../include/image_io.h"
#include "../include/gaussian.h"
#include "../include/sobel.h"

// ── GAUSSIAN TESTS ──────────────────────────────────────

TEST(GaussianTest, UniformImageInvariant) {
    image img = allocate_image(64, 64);
    for (int i = 0; i < 64 * 64; i++)
        img.data[i] = 128;
    image blurred = gaussian_blur(img);
    for (int r = 3; r < 61; r++)
        for (int c = 3; c < 61; c++)
            EXPECT_NEAR(blurred.data[r * 64 + c], 128, 1);
    free_image(img);
    free_image(blurred);
}

TEST(GaussianTest, AllBlackStaysBlack) {
    image img = allocate_image(64, 64);
    image blurred = gaussian_blur(img);
    for (int i = 0; i < 64 * 64; i++)
        EXPECT_EQ(blurred.data[i], 0);
    free_image(img);
    free_image(blurred);
}

TEST(GaussianTest, DimensionsPreserved) {
    image img = allocate_image(100, 75);
    image blurred = gaussian_blur(img);
    EXPECT_EQ(blurred.width,  100);
    EXPECT_EQ(blurred.height, 75);
    free_image(img);
    free_image(blurred);
}

TEST(GaussianTest, ImpulseResponse) {
    image img = allocate_image(64, 64);
    img.data[32 * 64 + 32] = 255;
    image blurred = gaussian_blur(img);
    EXPECT_GT(blurred.data[32 * 64 + 32], 0);
    EXPECT_GT(blurred.data[31 * 64 + 32], 0);
    EXPECT_GT(blurred.data[33 * 64 + 32], 0);
    EXPECT_GT(blurred.data[32 * 64 + 31], 0);
    EXPECT_GT(blurred.data[32 * 64 + 33], 0);
    free_image(img);
    free_image(blurred);
}

// ── SOBEL TESTS ──────────────────────────────────────────

TEST(SobelTest, ZeroGradientOnUniformImage) {
    image img = allocate_image(64, 64);
    for (int i = 0; i < 64 * 64; i++)
        img.data[i] = 128;
    sobel_result result = sobel_gradient(img);
    for (int r = 1; r < 63; r++)
        for (int c = 1; c < 63; c++)
            EXPECT_EQ(result.magnitude_l1.data[r * 64 + c], 0);
    free_image(img);
    free_sobel_result(result);
}

TEST(SobelTest, VerticalEdgeDetected) {
    image img = allocate_image(64, 64);
    for (int r = 0; r < 64; r++)
        for (int c = 32; c < 64; c++)
            img.data[r * 64 + c] = 255;
    sobel_result result = sobel_gradient(img);
    EXPECT_GT(result.magnitude_l1.data[32 * 64 + 32], 0);
    EXPECT_EQ(result.direction.data[32 * 64 + 32], 0);
    free_image(img);
    free_sobel_result(result);
}

TEST(SobelTest, HorizontalEdgeDetected) {
    image img = allocate_image(64, 64);
    for (int r = 32; r < 64; r++)
        for (int c = 0; c < 64; c++)
            img.data[r * 64 + c] = 255;
    sobel_result result = sobel_gradient(img);
    EXPECT_GT(result.magnitude_l1.data[32 * 64 + 32], 0);
    EXPECT_EQ(result.direction.data[32 * 64 + 32], 90);
    free_image(img);
    free_sobel_result(result);
}

TEST(SobelTest, BothMagnitudesNonZero) {
    image img = allocate_image(64, 64);
    for (int r = 20; r < 44; r++)
        for (int c = 20; c < 44; c++)
            img.data[r * 64 + c] = 255;
    sobel_result result = sobel_gradient(img);
    bool l1_nonzero = false;
    bool l2_nonzero = false;
    for (int i = 0; i < 64 * 64; i++) {
        if (result.magnitude_l1.data[i] > 0) l1_nonzero = true;
        if (result.magnitude_l2.data[i] > 0) l2_nonzero = true;
    }
    EXPECT_TRUE(l1_nonzero);
    EXPECT_TRUE(l2_nonzero);
    free_image(img);
    free_sobel_result(result);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
