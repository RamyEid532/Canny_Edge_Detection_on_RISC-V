#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../include/image_io.h"
#include "../include/gaussian.h"
#include "../include/sobel.h"
#include "../include/suppression.h"
#include "../include/thresholding.h"
#include "../include/hysteresis.h"


// =============================================================================
// TEST IMAGE GENERATORS
// Produces known synthetic patterns as `image` structs for use across suites.
// Each generator allocates — caller must free_image().
// =============================================================================
 
static image make_uniform(int W, int H, uint8_t val) {
    image img = allocate_image(W, H);
    memset(img.data, val, W * H);
    return img;
}
 
// Left half = 0, right half = 255
static image make_vertical_edge(int W, int H) {
    image img = allocate_image(W, H);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            img.data[r * W + c] = (c < W / 2) ? 0 : 255;
    return img;
}
 
// Top half = 0, bottom half = 255
static image make_horizontal_edge(int W, int H) {
    image img = allocate_image(W, H);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            img.data[r * W + c] = (r < H / 2) ? 0 : 255;
    return img;
}
 
// Upper-left triangle = 0, lower-right = 255
static image make_diagonal_edge(int W, int H) {
    image img = allocate_image(W, H);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            img.data[r * W + c] = (r + c < W) ? 0 : 255;
    return img;
}
 
// White rectangle on black background (good for all-four-directions testing)
static image make_rectangle(int W, int H) {
    image img = allocate_image(W, H);
    for (int r = H / 4; r < 3 * H / 4; r++)
        for (int c = W / 4; c < 3 * W / 4; c++)
            img.data[r * W + c] = 255;
    return img;
}
 
// Deterministic pseudo-random image (no rand() — fully reproducible)
static image make_random(int W, int H) {
    image img = allocate_image(W, H);
    for (int i = 0; i < W * H; i++)
        img.data[i] = (uint8_t)((i * 6364136223846793005ULL + 1442695040888963407ULL) >> 56);
    return img;
}






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

// Random image: must not crash, and output must stay in [0, 255]
TEST(SobelTest, RandomImageOutputInRange) {
    image img      = make_random(64, 64);
    sobel_result r = sobel_gradient(img);
    for (int i = 0; i < 64 * 64; i++) {
        EXPECT_GE(r.magnitude_l1.data[i], 0);
        EXPECT_LE(r.magnitude_l1.data[i], 255);
        EXPECT_GE(r.magnitude_l2.data[i], 0);
        EXPECT_LE(r.magnitude_l2.data[i], 255);
    }
    free_image(img);
    free_sobel_result(r);
}
 
// =============================================================================
// DIRECTION TESTS
// =============================================================================
 
// Vertical edge → Gx >> Gy → direction must be 0
TEST(DirectionTest, VerticalEdgeGivesDirection0) {
    image img      = make_vertical_edge(64, 64);
    sobel_result r = sobel_gradient(img);
    const int edge_col = 32;
    for (int row = 2; row < 62; row++)
        EXPECT_EQ(r.direction.data[row * 64 + edge_col], 0)
            << "Expected direction=0 at vertical edge, row=" << row;
    free_image(img);
    free_sobel_result(r);
}
 
// Horizontal edge → Gy >> Gx → direction must be 90
TEST(DirectionTest, HorizontalEdgeGivesDirection90) {
    image img      = make_horizontal_edge(64, 64);
    sobel_result r = sobel_gradient(img);
    const int edge_row = 32;
    for (int col = 2; col < 62; col++)
        EXPECT_EQ(r.direction.data[edge_row * 64 + col], 90)
            << "Expected direction=90 at horizontal edge, col=" << col;
    free_image(img);
    free_sobel_result(r);
}
 
// Diagonal edge → Gx ≈ Gy → direction must be 45 or 135
TEST(DirectionTest, DiagonalEdgeGivesDirection45or135) {
    image img      = make_diagonal_edge(64, 64);
    sobel_result r = sobel_gradient(img);
    bool found_diagonal = false;
    for (int row = 2; row < 62 && !found_diagonal; row++)
        for (int col = 2; col < 62 && !found_diagonal; col++) {
            int i       = row * 64 + col;
            uint8_t dir = r.direction.data[i];
            if (r.magnitude_l1.data[i] > 0 && (dir == 45 || dir == 135))
                found_diagonal = true;
        }
    EXPECT_TRUE(found_diagonal)
        << "Diagonal edge should produce direction=45 or 135 at some pixel";
    free_image(img);
    free_sobel_result(r);
}
 
// Direction values must always be one of the four legal angles
TEST(DirectionTest, AllDirectionValuesAreValid) {
    image img      = make_random(64, 64);
    sobel_result r = sobel_gradient(img);
    for (int i = 0; i < 64 * 64; i++) {
        uint8_t dir = r.direction.data[i];
        EXPECT_TRUE(dir == 0 || dir == 45 || dir == 90 || dir == 135)
            << "Direction value " << (int)dir << " is not a valid angle";
    }
    free_image(img);
    free_sobel_result(r);
}
 
// =============================================================================
// NON-MAXIMUM SUPPRESSION TESTS
// =============================================================================
 
TEST(SuppressionTest, OutputDimensionsMatch) {
    image mag = make_uniform(64, 64, 128);
    image dir = make_uniform(64, 64, 0);
    image out = non_maximum_suppression(mag, dir);
    EXPECT_EQ(out.width,  64);
    EXPECT_EQ(out.height, 64);
    free_image(mag); free_image(dir); free_image(out);
}
 
// Uniform magnitude: every pixel ties with its neighbors → all kept
TEST(SuppressionTest, UniformMagnitudeKeepsAllInteriorPixels) {
    image mag = make_uniform(64, 64, 200);
    image dir = make_uniform(64, 64, 0);
    image out = non_maximum_suppression(mag, dir);
    for (int r = 1; r < 63; r++)
        for (int c = 1; c < 63; c++)
            EXPECT_EQ(out.data[r * 64 + c], 200)
                << "Uniform image: interior pixel at (" << r << "," << c << ") should be kept";
    free_image(mag); free_image(dir); free_image(out);
}
 
// A lone bright peak surrounded by zeros must survive
TEST(SuppressionTest, SinglePeakSurvives) {
    const int W = 64, H = 64;
    image mag = allocate_image(W, H);
    image dir = make_uniform(W, H, 0);
    mag.data[32 * W + 32] = 255;
    image out = non_maximum_suppression(mag, dir);
    EXPECT_EQ(out.data[32 * W + 32], 255);
    free_image(mag); free_image(dir); free_image(out);
}
 
// Pixel weaker than its neighbour in the gradient direction must be suppressed
TEST(SuppressionTest, NonPeakPixelsSuppressed) {
    const int W = 64, H = 64;
    image mag = allocate_image(W, H);
    image dir = make_uniform(W, H, 0);   // compare left / right
    for (int r = 0; r < H; r++) {
        mag.data[r * W + 30] = 100;
        mag.data[r * W + 31] = 150;
        mag.data[r * W + 32] = 200;   // local peak
        mag.data[r * W + 33] = 150;
        mag.data[r * W + 34] = 100;
    }
    image out = non_maximum_suppression(mag, dir);
    for (int r = 1; r < H - 1; r++)
        EXPECT_EQ(out.data[r * W + 31], 0)
            << "Non-peak at col 31 should be suppressed, row=" << r;
    free_image(mag); free_image(dir); free_image(out);
}
 
// Border pixels are left at 0 by the implementation (it skips row/col 0 and last)
TEST(SuppressionTest, BorderPixelsAreZero) {
    const int W = 64, H = 64;
    image mag = make_uniform(W, H, 200);
    image dir = make_uniform(W, H, 0);
    image out = non_maximum_suppression(mag, dir);
    for (int c = 0; c < W; c++) {
        EXPECT_EQ(out.data[0 * W + c],       0);
        EXPECT_EQ(out.data[(H-1) * W + c],   0);
    }
    for (int r = 0; r < H; r++) {
        EXPECT_EQ(out.data[r * W + 0],       0);
        EXPECT_EQ(out.data[r * W + (W-1)],   0);
    }
    free_image(mag); free_image(dir); free_image(out);
}
 
// All four direction codes must produce valid output without crashing
TEST(SuppressionTest, AllDirectionsHandled) {
    const int W = 16, H = 16;
    for (uint8_t dv : {(uint8_t)0, (uint8_t)45, (uint8_t)90, (uint8_t)135}) {
        image mag = make_uniform(W, H, 128);
        image dir = make_uniform(W, H, dv);
        image out = non_maximum_suppression(mag, dir);
        for (int i = 0; i < W * H; i++)
            EXPECT_LE(out.data[i], 255);
        free_image(mag); free_image(dir); free_image(out);
    }
}
 
// =============================================================================
// DOUBLE THRESHOLD TESTS
// =============================================================================
 
TEST(ThresholdTest, StrongEdgesClassifiedCorrectly) {
    image nms = make_uniform(16, 16, 200);   // all above high=100
    image out = double_threshold(nms, 50, 100);
    for (int i = 0; i < 16 * 16; i++)
        EXPECT_EQ(out.data[i], STRONG_EDGE);
    free_image(nms); free_image(out);
}
 
TEST(ThresholdTest, WeakEdgesClassifiedCorrectly) {
    image nms = make_uniform(16, 16, 70);    // in [low=50, high=100)
    image out = double_threshold(nms, 50, 100);
    for (int i = 0; i < 16 * 16; i++)
        EXPECT_EQ(out.data[i], WEAK_EDGE);
    free_image(nms); free_image(out);
}
 
TEST(ThresholdTest, NoEdgesClassifiedCorrectly) {
    image nms = make_uniform(16, 16, 20);    // below low=50
    image out = double_threshold(nms, 50, 100);
    for (int i = 0; i < 16 * 16; i++)
        EXPECT_EQ(out.data[i], NO_EDGE);
    free_image(nms); free_image(out);
}
 
TEST(ThresholdTest, ThreeClassesPresentInMixedImage) {
    image nms   = allocate_image(3, 1);
    nms.data[0] = 20;    // → NO_EDGE
    nms.data[1] = 70;    // → WEAK_EDGE
    nms.data[2] = 200;   // → STRONG_EDGE
    image out = double_threshold(nms, 50, 100);
    EXPECT_EQ(out.data[0], NO_EDGE);
    EXPECT_EQ(out.data[1], WEAK_EDGE);
    EXPECT_EQ(out.data[2], STRONG_EDGE);
    free_image(nms); free_image(out);
}
 
TEST(ThresholdTest, OutputDimensionsMatch) {
    image nms = make_uniform(100, 75, 128);
    image out = double_threshold(nms, 50, 150);
    EXPECT_EQ(out.width,  100);
    EXPECT_EQ(out.height, 75);
    free_image(nms); free_image(out);
}
 
// With low == high there is no weak band: every pixel is strong or absent
TEST(ThresholdTest, NoWeakPixelsWhenLowEqualsHigh) {
    image nms = make_random(64, 64);
    image out = double_threshold(nms, 128, 128);
    for (int i = 0; i < 64 * 64; i++) {
        uint8_t v = out.data[i];
        EXPECT_TRUE(v == STRONG_EDGE || v == NO_EDGE)
            << "With low==high, WEAK_EDGE should not appear";
    }
    free_image(nms); free_image(out);
}
 
// =============================================================================
// HYSTERESIS TESTS
// =============================================================================
 
TEST(HysteresisTest, StrongEdgesAlwaysSurvive) {
    const int W = 64, H = 64;
    image thr = make_uniform(W, H, STRONG_EDGE);
    image out = hysteresis_tracking(thr);
    for (int r = 1; r < H - 1; r++)
        for (int c = 1; c < W - 1; c++)
            EXPECT_EQ(out.data[r * W + c], STRONG_EDGE)
                << "Strong edge at (" << r << "," << c << ") should survive";
    free_image(thr); free_image(out);
}
 
TEST(HysteresisTest, IsolatedWeakEdgesAreRemoved) {
    const int W = 16, H = 16;
    image thr = allocate_image(W, H);   // all NO_EDGE
    thr.data[8 * W + 8] = WEAK_EDGE;
    image out = hysteresis_tracking(thr);
    EXPECT_EQ(out.data[8 * W + 8], NO_EDGE)
        << "Isolated weak edge must be removed";
    free_image(thr); free_image(out);
}
 
TEST(HysteresisTest, WeakEdgeAdjacentToStrongSurvives) {
    const int W = 16, H = 16;
    image thr = allocate_image(W, H);
    thr.data[8 * W + 8] = STRONG_EDGE;
    thr.data[8 * W + 9] = WEAK_EDGE;    // directly to the right
    image out = hysteresis_tracking(thr);
    EXPECT_EQ(out.data[8 * W + 9], STRONG_EDGE)
        << "Weak edge adjacent to strong should become STRONG_EDGE";
    free_image(thr); free_image(out);
}
 
TEST(HysteresisTest, WeakEdgeDiagonallyConnectedToStrongSurvives) {
    const int W = 16, H = 16;
    image thr = allocate_image(W, H);
    thr.data[7 * W + 7] = STRONG_EDGE;
    thr.data[8 * W + 8] = WEAK_EDGE;    // diagonal neighbor
    image out = hysteresis_tracking(thr);
    EXPECT_EQ(out.data[8 * W + 8], STRONG_EDGE)
        << "Diagonally connected weak edge should survive (8-connectivity)";
    free_image(thr); free_image(out);
}
 
TEST(HysteresisTest, NoEdgePixelsStayZero) {
    const int W = 16, H = 16;
    image thr = allocate_image(W, H);
    // Surround center with strong edges but keep center as NO_EDGE
    for (int r = 6; r <= 10; r++)
        for (int c = 6; c <= 10; c++)
            thr.data[r * W + c] = STRONG_EDGE;
    thr.data[8 * W + 8] = NO_EDGE;
    image out = hysteresis_tracking(thr);
    EXPECT_EQ(out.data[8 * W + 8], NO_EDGE)
        << "NO_EDGE pixel must not become an edge even when surrounded by strong edges";
    free_image(thr); free_image(out);
}
 
TEST(HysteresisTest, OutputDimensionsMatch) {
    image thr = make_uniform(100, 75, 0);
    image out = hysteresis_tracking(thr);
    EXPECT_EQ(out.width,  100);
    EXPECT_EQ(out.height, 75);
    free_image(thr); free_image(out);
}
 
TEST(HysteresisTest, OutputContainsOnlyValidValues) {
    const int W = 64, H = 64;
    image thr = make_random(W, H);
    for (int i = 0; i < W * H; i++) {
        uint8_t v = thr.data[i];
        thr.data[i] = (v > 200) ? STRONG_EDGE : (v > 100) ? WEAK_EDGE : NO_EDGE;
    }
    image out = hysteresis_tracking(thr);
    for (int i = 0; i < W * H; i++) {
        uint8_t v = out.data[i];
        EXPECT_TRUE(v == STRONG_EDGE || v == WEAK_EDGE || v == NO_EDGE)
            << "Unexpected output value: " << (int)v;
    }
    free_image(thr); free_image(out);
}
 

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}