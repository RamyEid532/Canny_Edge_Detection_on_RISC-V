// =========================================================================
// thresholding_rvv.cpp
// RVV-accelerated Double Thresholding
// Note: 0.8% of runtime — vectorized for completeness
// =========================================================================
#include "thresholding.h"
#include <riscv_vector.h>

image double_threshold_rvv(
    const image& nms_image,
    uint8_t low_threshold,
    uint8_t high_threshold)
{
    image output = allocate_image(nms_image.width, nms_image.height);

    int size = nms_image.width * nms_image.height;
    int i = 0;

    while (i < size) {
        // set vl for u8, LMUL=1
        // VLEN=128 → vl=16, VLEN=256 → vl=32, VLEN=512 → vl=64
        size_t vl = __riscv_vsetvl_e8m1(size - i);

        // __riscv_vle8_v_u8m1: load vl pixels
        vuint8m1_t vpix = __riscv_vle8_v_u8m1(nms_image.data + i, vl);

        // __riscv_vmsgeu_vx_u8m1_b8: mask where pixel >= high → STRONG
        vbool8_t strong_mask = __riscv_vmsgeu_vx_u8m1_b8(vpix, high_threshold, vl);

        // __riscv_vmsgeu_vx_u8m1_b8: mask where pixel >= low → WEAK candidate
        vbool8_t weak_mask = __riscv_vmsgeu_vx_u8m1_b8(vpix, low_threshold, vl);

        // weak = (pixel >= low) AND NOT (pixel >= high)
        // __riscv_vmandn_mm_b8: weak_mask AND NOT strong_mask
        vbool8_t only_weak = __riscv_vmandn_mm_b8(weak_mask, strong_mask, vl);

        // Start with NO_EDGE
        vuint8m1_t vresult = __riscv_vmv_v_x_u8m1(NO_EDGE, vl);

        // Apply WEAK_EDGE where only_weak
        // __riscv_vmerge_vxm_u8m1: merge scalar into vector using mask
        vresult = __riscv_vmerge_vxm_u8m1(vresult, WEAK_EDGE, only_weak, vl);

        // Apply STRONG_EDGE where strong_mask
        vresult = __riscv_vmerge_vxm_u8m1(vresult, STRONG_EDGE, strong_mask, vl);

        // __riscv_vse8_v_u8m1: store result
        __riscv_vse8_v_u8m1(output.data + i, vresult, vl);

        i += vl;
    }

    return output;
}
