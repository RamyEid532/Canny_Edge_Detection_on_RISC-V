#include "sobel.h"
#include <riscv_vector.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>

void compute_gxgy_rvv(const image& input, int16_t* Gx, int16_t* Gy) {
    int W = input.width;
    int H = input.height;

    for (int row = 1; row < H - 1; row++) {
        int col = 1;
        while (col < W - 1) {
            size_t vl = __riscv_vsetvl_e32m4(W - 1 - col);

            vint32m4_t vgx = __riscv_vmv_v_x_i32m4(0, vl);
            vint32m4_t vgy = __riscv_vmv_v_x_i32m4(0, vl);

            // Unroll kernel manually — أسرع من loop
            // كل coefficient معروف مسبقاً — مش محتاج if checks

            // Row -1
            {
                // (-1,-1): coeff_x=-1, coeff_y=-1
                const uint8_t* s = input.data + (row-1)*W + (col-1);
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgx = __riscv_vsub_vv_i32m4(vgx, pi32, vl);
                vgy = __riscv_vsub_vv_i32m4(vgy, pi32, vl);
            }
            {
                // (-1,0): coeff_x=0, coeff_y=-2
                const uint8_t* s = input.data + (row-1)*W + col;
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgy = __riscv_vmacc_vx_i32m4(vgy, -2, pi32, vl);
            }
            {
                // (-1,1): coeff_x=1, coeff_y=-1
                const uint8_t* s = input.data + (row-1)*W + (col+1);
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgx = __riscv_vadd_vv_i32m4(vgx, pi32, vl);
                vgy = __riscv_vsub_vv_i32m4(vgy, pi32, vl);
            }
            // Row 0
            {
                // (0,-1): coeff_x=-2, coeff_y=0
                const uint8_t* s = input.data + row*W + (col-1);
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgx = __riscv_vmacc_vx_i32m4(vgx, -2, pi32, vl);
            }
            {
                // (0,1): coeff_x=2, coeff_y=0
                const uint8_t* s = input.data + row*W + (col+1);
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgx = __riscv_vmacc_vx_i32m4(vgx, 2, pi32, vl);
            }
            // Row +1
            {
                // (1,-1): coeff_x=-1, coeff_y=1
                const uint8_t* s = input.data + (row+1)*W + (col-1);
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgx = __riscv_vsub_vv_i32m4(vgx, pi32, vl);
                vgy = __riscv_vadd_vv_i32m4(vgy, pi32, vl);
            }
            {
                // (1,0): coeff_x=0, coeff_y=2
                const uint8_t* s = input.data + (row+1)*W + col;
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgy = __riscv_vmacc_vx_i32m4(vgy, 2, pi32, vl);
            }
            {
                // (1,1): coeff_x=1, coeff_y=1
                const uint8_t* s = input.data + (row+1)*W + (col+1);
                vuint8m1_t p = __riscv_vle8_v_u8m1(s, vl);
                vuint16m2_t p16 = __riscv_vwcvtu_x_x_v_u16m2(p, vl);
                vint32m4_t pi32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vwcvtu_x_x_v_u32m4(p16, vl));
                vgx = __riscv_vadd_vv_i32m4(vgx, pi32, vl);
                vgy = __riscv_vadd_vv_i32m4(vgy, pi32, vl);
            }

            vint16m2_t vgx16 = __riscv_vnclip_wx_i16m2(vgx, 0, 0, vl);
            vint16m2_t vgy16 = __riscv_vnclip_wx_i16m2(vgy, 0, 0, vl);
            __riscv_vse16_v_i16m2(Gx + row*W + col, vgx16, vl);
            __riscv_vse16_v_i16m2(Gy + row*W + col, vgy16, vl);

            col += vl;
        }
    }

    // Border scalar fallback
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            if (row >= 1 && row < H-1 && col >= 1 && col < W-1) continue;
            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int r = row+ky, c = col+kx;
                    uint8_t pixel = 0;
                    if (r >= 0 && r < H && c >= 0 && c < W)
                        pixel = input.data[r*W+c];
                    gx += pixel * SOBEL_X[ky+1][kx+1];
                    gy += pixel * SOBEL_Y[ky+1][kx+1];
                }
            }
            Gx[row*W+col] = (int16_t)gx;
            Gy[row*W+col] = (int16_t)gy;
        }
    }
}

void compute_magnitude_rvv(int16_t* Gx, int16_t* Gy, int size,
                            image& mag_l1, image& mag_l2) {

    int max_l1 = 1;
    int i = 0;
    vint16m1_t vmax_scalar = __riscv_vmv_v_x_i16m1(0, 1);

    while (i < size) {
        size_t vl = __riscv_vsetvl_e16m4(size - i);
        vint16m4_t vgx = __riscv_vle16_v_i16m4(Gx + i, vl);
        vint16m4_t vgy = __riscv_vle16_v_i16m4(Gy + i, vl);
        vint16m4_t vax = __riscv_vmax_vv_i16m4(vgx, __riscv_vneg_v_i16m4(vgx, vl), vl);
        vint16m4_t vay = __riscv_vmax_vv_i16m4(vgy, __riscv_vneg_v_i16m4(vgy, vl), vl);
        vint16m4_t vl1 = __riscv_vadd_vv_i16m4(vax, vay, vl);
        vmax_scalar = __riscv_vredmax_vs_i16m4_i16m1(vl1, vmax_scalar, vl);
        i += vl;
    }

    max_l1 = std::max((int)__riscv_vmv_x_s_i16m1_i16(vmax_scalar), 1);

    // Fixed-point بدل division — أسرع بكتير
    // scale = (255 * 65536) / max_l1
    int32_t scale = (255 * 65536) / max_l1;

    i = 0;
    while (i < size) {
        size_t vl = __riscv_vsetvl_e16m4(size - i);
        vint16m4_t vgx = __riscv_vle16_v_i16m4(Gx + i, vl);
        vint16m4_t vgy = __riscv_vle16_v_i16m4(Gy + i, vl);
        vint16m4_t vax = __riscv_vmax_vv_i16m4(vgx, __riscv_vneg_v_i16m4(vgx, vl), vl);
        vint16m4_t vay = __riscv_vmax_vv_i16m4(vgy, __riscv_vneg_v_i16m4(vgy, vl), vl);
        vint16m4_t vl1 = __riscv_vadd_vv_i16m4(vax, vay, vl);

        // Widen i16m4 → i32m8
        vint32m8_t vl1_wide = __riscv_vwmul_vx_i32m8(vl1, (int16_t)(scale >> 8), vl);

        // Right shift بدل division
        vuint32m8_t vl1_u = __riscv_vreinterpret_v_i32m8_u32m8(vl1_wide);
        vl1_u = __riscv_vsrl_vx_u32m8(vl1_u, 8, vl);
        vint32m8_t vl1_norm = __riscv_vreinterpret_v_u32m8_i32m8(vl1_u);

        // Narrow i32m8 → i16m4 → u8m2
        vint16m4_t vnarrow16 = __riscv_vnclip_wx_i16m4(vl1_norm, 0, 0, vl);
        vuint8m2_t vresult = __riscv_vnclipu_wx_u8m2(
            __riscv_vreinterpret_v_i16m4_u16m4(vnarrow16), 0, 0, vl);

        __riscv_vse8_v_u8m2(mag_l1.data + i, vresult, vl);
        i += vl;
    }

    // L2 scalar
    float max_l2 = 1.0f;
    for (int j = 0; j < size; j++) {
        float l2 = std::sqrt((float)Gx[j]*Gx[j] + (float)Gy[j]*Gy[j]);
        if (l2 > max_l2) max_l2 = l2;
    }
    for (int j = 0; j < size; j++) {
        float l2 = std::sqrt((float)Gx[j]*Gx[j] + (float)Gy[j]*Gy[j]);
        mag_l2.data[j] = (uint8_t)(l2 * 255.0f / max_l2);
    }
}

void compute_direction_rvv(int16_t* Gx, int16_t* Gy, int size, image& direction) {
    for (int i = 0; i < size; i++) {
        int ax = std::abs(Gx[i]);
        int ay = std::abs(Gy[i]);
        uint8_t dir;
        if      (ay * 5 < ax * 2)  dir = 0;
        else if (ay * 5 < ax * 12) dir = 45;
        else if (ax * 5 < ay * 2)  dir = 90;
        else                       dir = 135;
        direction.data[i] = dir;
    }
}
