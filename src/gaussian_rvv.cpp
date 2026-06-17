#include "gaussian.h"
#include <riscv_vector.h>
#include <cstdlib>
#include <cstring>

image gaussian_blur_rvv(const image& input) {
    const int W = input.width;
    const int H = input.height;
    image output = allocate_image(W, H);

    for (int row = 0; row < H; row++) {
        // Border rows: scalar
        if (row < 2 || row >= H-2) {
            for (int col = 0; col < W; col++) {
                int32_t sum = 0;
                for (int ky = -2; ky <= 2; ky++)
                    for (int kx = -2; kx <= 2; kx++) {
                        int r = row+ky, c = col+kx;
                        uint8_t p = (r>=0&&r<H&&c>=0&&c<W) ? input.data[r*W+c] : 0;
                        sum += (int32_t)p * GAUSS_KERNEL[ky+2][kx+2];
                    }
                int32_t res = sum / GAUSS_SUM;
                output.data[row*W+col] = (uint8_t)(res>255?255:res<0?0:res);
            }
            continue;
        }

        // Border columns: scalar
        for (int col = 0; col < 2; col++) {
            int32_t sum = 0;
            for (int ky=-2;ky<=2;ky++)
                for (int kx=-2;kx<=2;kx++) {
                    int r=row+ky, c=col+kx;
                    uint8_t p=(r>=0&&r<H&&c>=0&&c<W)?input.data[r*W+c]:0;
                    sum+=(int32_t)p*GAUSS_KERNEL[ky+2][kx+2];
                }
            int32_t res=sum/GAUSS_SUM;
            output.data[row*W+col]=(uint8_t)(res>255?255:res<0?0:res);
        }
        for (int col = W-2; col < W; col++) {
            int32_t sum = 0;
            for (int ky=-2;ky<=2;ky++)
                for (int kx=-2;kx<=2;kx++) {
                    int r=row+ky, c=col+kx;
                    uint8_t p=(r>=0&&r<H&&c>=0&&c<W)?input.data[r*W+c]:0;
                    sum+=(int32_t)p*GAUSS_KERNEL[ky+2][kx+2];
                }
            int32_t res=sum/GAUSS_SUM;
            output.data[row*W+col]=(uint8_t)(res>255?255:res<0?0:res);
        }

        // Interior: pure RVV - no boundary checks
        int col = 2;
        while (col < W-2) {
            size_t vl = __riscv_vsetvl_e32m4((W-2) - col);
            vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int kern = GAUSS_KERNEL[ky+2][kx+2];
                    if (kern == 0) continue;
                    const uint8_t* src = &input.data[(row+ky)*W + col+kx];
                    vuint8m1_t  vu8  = __riscv_vle8_v_u8m1(src, vl);
                    vuint16m2_t vu16 = __riscv_vzext_vf2_u16m2(vu8, vl);
                    vint32m4_t  vi32 = __riscv_vreinterpret_v_u32m4_i32m4(
                                        __riscv_vzext_vf2_u32m4(vu16, vl));
                    vacc = __riscv_vmacc_vx_i32m4(vacc, kern, vi32, vl);
                }
            }

            vacc = __riscv_vdiv_vx_i32m4(vacc, GAUSS_SUM, vl);
            vacc = __riscv_vmax_vx_i32m4(vacc, 0, vl);
            vacc = __riscv_vmin_vx_i32m4(vacc, 255, vl);
            vint16m2_t vi16 = __riscv_vnsra_wx_i16m2(vacc, 0, vl);
            vuint8m1_t vu8  = __riscv_vnclipu_wx_u8m1(
                               __riscv_vreinterpret_v_i16m2_u16m2(vi16), 0, 0, vl);
            __riscv_vse8_v_u8m1(&output.data[row*W+col], vu8, vl);
            col += vl;
        }
    }
    return output;
}
