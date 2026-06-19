// =========================================================================
// suppression_rvv.cpp
// RVV-accelerated Non-Maximum Suppression
// Note: NMS is 3.6% of runtime — speedup impact is minimal (Amdahl's law)
// Vectorized for completeness and demonstration of RVV mastery
// =========================================================================
#include "suppression.h"
#include <riscv_vector.h>

image non_maximum_suppression_rvv(
    const image& magnitude,
    const image& direction)
{
    image output = allocate_image(magnitude.width, magnitude.height);

    int W = magnitude.width;
    int H = magnitude.height;

    // Border pixels stay 0 (already zeroed by allocate_image)
    for (int y = 1; y < H - 1; y++) {
        int x = 1;
        while (x < W - 1) {
            // set vl for u8, LMUL=1
            // VLEN=128 → vl=16, VLEN=256 → vl=32, VLEN=512 → vl=64
            size_t vl = __riscv_vsetvl_e8m1(W - 1 - x);

            // Load current row pixels
            // __riscv_vle8_v_u8m1: load vl bytes from magnitude
            vuint8m1_t vcurrent = __riscv_vle8_v_u8m1(
                magnitude.data + y * W + x, vl);

            // Load direction
            vuint8m1_t vdir = __riscv_vle8_v_u8m1(
                direction.data + y * W + x, vl);

            // Load all neighbors for all directions
            // dir=0: left/right
            vuint8m1_t vn1_0 = __riscv_vle8_v_u8m1(
                magnitude.data + y * W + x - 1, vl);
            vuint8m1_t vn2_0 = __riscv_vle8_v_u8m1(
                magnitude.data + y * W + x + 1, vl);

            // dir=90: up/down
            vuint8m1_t vn1_90 = __riscv_vle8_v_u8m1(
                magnitude.data + (y-1) * W + x, vl);
            vuint8m1_t vn2_90 = __riscv_vle8_v_u8m1(
                magnitude.data + (y+1) * W + x, vl);

            // dir=45: upper-right/lower-left
            vuint8m1_t vn1_45 = __riscv_vle8_v_u8m1(
                magnitude.data + (y-1) * W + x + 1, vl);
            vuint8m1_t vn2_45 = __riscv_vle8_v_u8m1(
                magnitude.data + (y+1) * W + x - 1, vl);

            // dir=135: upper-left/lower-right
            vuint8m1_t vn1_135 = __riscv_vle8_v_u8m1(
                magnitude.data + (y-1) * W + x - 1, vl);
            vuint8m1_t vn2_135 = __riscv_vle8_v_u8m1(
                magnitude.data + (y+1) * W + x + 1, vl);

            // Select neighbors based on direction using masks
            // __riscv_vmseq_vx_u8m1_b8: mask where dir == value
            vbool8_t mask0   = __riscv_vmseq_vx_u8m1_b8(vdir, 0,   vl);
            vbool8_t mask45  = __riscv_vmseq_vx_u8m1_b8(vdir, 45,  vl);
            vbool8_t mask90  = __riscv_vmseq_vx_u8m1_b8(vdir, 90,  vl);
            vbool8_t mask135 = __riscv_vmseq_vx_u8m1_b8(vdir, 135, vl);

            // Build neighbor1 and neighbor2 by merging with masks
            vuint8m1_t vneighbor1 = __riscv_vmv_v_x_u8m1(0, vl);
            vuint8m1_t vneighbor2 = __riscv_vmv_v_x_u8m1(0, vl);

            // __riscv_vmerge_vvm_u8m1: select from two vectors using mask
            vneighbor1 = __riscv_vmerge_vvm_u8m1(vneighbor1, vn1_0,   mask0,   vl);
            vneighbor1 = __riscv_vmerge_vvm_u8m1(vneighbor1, vn1_45,  mask45,  vl);
            vneighbor1 = __riscv_vmerge_vvm_u8m1(vneighbor1, vn1_90,  mask90,  vl);
            vneighbor1 = __riscv_vmerge_vvm_u8m1(vneighbor1, vn1_135, mask135, vl);

            vneighbor2 = __riscv_vmerge_vvm_u8m1(vneighbor2, vn2_0,   mask0,   vl);
            vneighbor2 = __riscv_vmerge_vvm_u8m1(vneighbor2, vn2_45,  mask45,  vl);
            vneighbor2 = __riscv_vmerge_vvm_u8m1(vneighbor2, vn2_90,  mask90,  vl);
            vneighbor2 = __riscv_vmerge_vvm_u8m1(vneighbor2, vn2_135, mask135, vl);

            // Suppress: current >= neighbor1 && current >= neighbor2
            // __riscv_vmsgeu_vv_u8m1_b8: mask where current >= neighbor
            vbool8_t keep1 = __riscv_vmsgeu_vv_u8m1_b8(vcurrent, vneighbor1, vl);
            vbool8_t keep2 = __riscv_vmsgeu_vv_u8m1_b8(vcurrent, vneighbor2, vl);

            // __riscv_vmand_mm_b8: AND of two masks
            vbool8_t keep = __riscv_vmand_mm_b8(keep1, keep2, vl);

            // Apply mask: keep current if local max, else 0
            // __riscv_vmerge_vxm_u8m1: select current or 0 based on mask
            vuint8m1_t vresult = __riscv_vmerge_vxm_u8m1(vcurrent, 0, keep, vl);

            // Wait — vmerge keeps second operand where mask=1
            // We want: keep=1 → current, keep=0 → 0
            vuint8m1_t vzero = __riscv_vmv_v_x_u8m1(0, vl);
            vresult = __riscv_vmerge_vvm_u8m1(vzero, vcurrent, keep, vl);

            // __riscv_vse8_v_u8m1: store result
            __riscv_vse8_v_u8m1(output.data + y * W + x, vresult, vl);

            x += vl;
        }
    }

    return output;
}
