#include "gaussian.h"
#include <riscv_vector.h>
#include <cstdlib>

image gaussian_blur_rvv(const image& input) {
    const int W = input.width;
    const int H = input.height;
    image output = allocate_image(W, H);

    for (int row = 0; row < H; row++) {
        // --- BORDER COLUMNS: scalar fallback (first 2 and last 2 cols) ---
        for (int col = 0; col < W; col++) {
            bool is_border = (row < 2 || row >= H-2 || col < 2 || col >= W-2);
            if (!is_border) continue;
            int32_t sum = 0;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int r = row + ky, c = col + kx;
                    uint8_t pixel = 0;
                    if (r >= 0 && r < H && c >= 0 && c < W)
                        pixel = input.data[r * W + c];
                    sum += (int32_t)pixel * GAUSS_KERNEL[ky+2][kx+2];
                }
            }
            int32_t res = sum / GAUSS_SUM;
            if (res > 255) res = 255;
            if (res < 0)   res = 0;
            output.data[row * W + col] = (uint8_t)res;
        }

        // --- INTERIOR: pure RVV (no boundary checks needed) ---
        if (row < 2 || row >= H-2) continue;

        int col = 2;
        while (col < W-2) {
            int remaining = (W-2) - col;
            size_t vl = __riscv_vsetvl_e32m4(remaining);

            vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int kern_val = GAUSS_KERNEL[ky+2][kx+2];
                    if (kern_val == 0) continue;

                    int r = row + ky;
                    // No boundary check needed - interior pixels guaranteed valid
                    const uint8_t* src = &input.data[r * W + col + kx];

                    vuint8m1_t  vpix_u8  = __riscv_vle8_v_u8m1(src, vl);
                    vuint16m2_t vpix_u16 = __riscv_vzext_vf2_u16m2(vpix_u8, vl);
                    vint32m4_t  vpix_i32 = __riscv_vreinterpret_v_u32m4_i32m4(
                                    __riscv_vzext_vf2_u32m4(vpix_u16, vl));

                    vacc = __riscv_vmacc_vx_i32m4(vacc, (int32_t)kern_val, vpix_i32, vl);
                }
            }

            vacc = __riscv_vdiv_vx_i32m4(vacc, GAUSS_SUM, vl);
            vacc = __riscv_vmax_vx_i32m4(vacc, 0,   vl);
            vacc = __riscv_vmin_vx_i32m4(vacc, 255, vl);

            vint16m2_t vout_i16 = __riscv_vnsra_wx_i16m2(vacc, 0, vl);
            vuint8m1_t vout_u8  = __riscv_vnclipu_wx_u8m1(
                                    __riscv_vreinterpret_v_i16m2_u16m2(vout_i16), 0, 0, vl);
            __riscv_vse8_v_u8m1(&output.data[row * W + col], vout_u8, vl);

            col += vl;
        }
    }
    return output;
}
