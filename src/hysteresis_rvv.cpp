// =========================================================================
// hysteresis_rvv.cpp
// RVV-accelerated Hysteresis — partial vectorization
// Note: 1.7% of runtime — DFS loop has data dependencies, hard to vectorize
// Vectorized the initial copy and final cleanup passes only
// =========================================================================
#include "hysteresis.h"
#include <riscv_vector.h>
#include <vector>

image hysteresis_tracking_rvv(const image& thresholded) {
    image output = allocate_image(thresholded.width, thresholded.height);

    int W = thresholded.width;
    int H = thresholded.height;
    int size = W * H;

    // -------------------------------------------------------
    // Pass 1: vectorized copy
    // -------------------------------------------------------
    int i = 0;
    while (i < size) {
        // set vl for u8, LMUL=1
        size_t vl = __riscv_vsetvl_e8m1(size - i);

        // __riscv_vle8_v_u8m1: load vl bytes
        vuint8m1_t vpix = __riscv_vle8_v_u8m1(thresholded.data + i, vl);

        // __riscv_vse8_v_u8m1: store vl bytes
        __riscv_vse8_v_u8m1(output.data + i, vpix, vl);

        i += vl;
    }

    // -------------------------------------------------------
    // DFS: scalar (data dependencies prevent vectorization)
    // -------------------------------------------------------
    std::vector<int> stack;
    stack.reserve(size / 4);

    for (int j = 0; j < size; j++)
        if (output.data[j] == STRONG_EDGE)
            stack.push_back(j);

    while (!stack.empty()) {
        int idx = stack.back();
        stack.pop_back();
        int y = idx / W;
        int x = idx % W;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (y+dy < 0 || y+dy >= H || x+dx < 0 || x+dx >= W) continue;
                int nidx = (y+dy)*W + (x+dx);
                if (output.data[nidx] == WEAK_EDGE) {
                    output.data[nidx] = STRONG_EDGE;
                    stack.push_back(nidx);
                }
            }
        }
    }

    // -------------------------------------------------------
    // Pass 3: vectorized cleanup — remove remaining weak edges
    // -------------------------------------------------------
    i = 0;
    while (i < size) {
        size_t vl = __riscv_vsetvl_e8m1(size - i);

        vuint8m1_t vpix = __riscv_vle8_v_u8m1(output.data + i, vl);

        // mask where pixel == WEAK_EDGE
        // __riscv_vmseq_vx_u8m1_b8: compare each element to WEAK_EDGE
        vbool8_t weak_mask = __riscv_vmseq_vx_u8m1_b8(vpix, WEAK_EDGE, vl);

        // set weak pixels to NO_EDGE
        // __riscv_vmerge_vxm_u8m1: replace with NO_EDGE where mask=1
        vpix = __riscv_vmerge_vxm_u8m1(vpix, NO_EDGE, weak_mask, vl);

        __riscv_vse8_v_u8m1(output.data + i, vpix, vl);

        i += vl;
    }

    return output;
}
