#include "sobel.h"
#include <riscv_vector.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>

sobel_result sobel_gradient_rvv(const image& input) {
    int W = input.width;
    int H = input.height;

    sobel_result result;
    result.magnitude_l1 = allocate_image(W, H);
    result.magnitude_l2 = allocate_image(W, H);
    result.direction    = allocate_image(W, H);
    result.Gx = new int16_t[W * H]();
    result.Gy = new int16_t[W * H]();

    const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    for (int row = 0; row < H; row++) {
        // Border rows: scalar
        if (row < 1 || row >= H-1) {
            for (int col = 0; col < W; col++) {
                int gx=0, gy=0;
                for (int ky=-1;ky<=1;ky++)
                    for (int kx=-1;kx<=1;kx++) {
                        int r=row+ky, c=col+kx;
                        uint8_t p=(r>=0&&r<H&&c>=0&&c<W)?input.data[r*W+c]:0;
                        gx+=p*KX[ky+1][kx+1];
                        gy+=p*KY[ky+1][kx+1];
                    }
                result.Gx[row*W+col]=(int16_t)gx;
                result.Gy[row*W+col]=(int16_t)gy;
            }
            continue;
        }

        // Border columns: scalar
        for (int col = 0; col < 1; col++) {
            int gx=0,gy=0;
            for (int ky=-1;ky<=1;ky++)
                for (int kx=-1;kx<=1;kx++) {
                    int r=row+ky,c=col+kx;
                    uint8_t p=(r>=0&&r<H&&c>=0&&c<W)?input.data[r*W+c]:0;
                    gx+=p*KX[ky+1][kx+1]; gy+=p*KY[ky+1][kx+1];
                }
            result.Gx[row*W+col]=(int16_t)gx;
            result.Gy[row*W+col]=(int16_t)gy;
        }
        for (int col = W-1; col < W; col++) {
            int gx=0,gy=0;
            for (int ky=-1;ky<=1;ky++)
                for (int kx=-1;kx<=1;kx++) {
                    int r=row+ky,c=col+kx;
                    uint8_t p=(r>=0&&r<H&&c>=0&&c<W)?input.data[r*W+c]:0;
                    gx+=p*KX[ky+1][kx+1]; gy+=p*KY[ky+1][kx+1];
                }
            result.Gx[row*W+col]=(int16_t)gx;
            result.Gy[row*W+col]=(int16_t)gy;
        }

        // Interior: pure RVV
        int col = 1;
        while (col < W-1) {
            size_t vl = __riscv_vsetvl_e16m2((W-1) - col);
            vint16m2_t vgx = __riscv_vmv_v_x_i16m2(0, vl);
            vint16m2_t vgy = __riscv_vmv_v_x_i16m2(0, vl);

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int kval_x = KX[ky+1][kx+1];
                    int kval_y = KY[ky+1][kx+1];
                    if (kval_x==0 && kval_y==0) continue;

                    const uint8_t* src = &input.data[(row+ky)*W + col+kx];
                    vuint8m1_t  vu8  = __riscv_vle8_v_u8m1(src, vl);
                    vuint16m2_t vu16 = __riscv_vzext_vf2_u16m2(vu8, vl);
                    vint16m2_t  vi16 = __riscv_vreinterpret_v_u16m2_i16m2(vu16);

                    if (kval_x != 0)
                        vgx = __riscv_vmacc_vx_i16m2(vgx, (int16_t)kval_x, vi16, vl);
                    if (kval_y != 0)
                        vgy = __riscv_vmacc_vx_i16m2(vgy, (int16_t)kval_y, vi16, vl);
                }
            }

            __riscv_vse16_v_i16m2(&result.Gx[row*W+col], vgx, vl);
            __riscv_vse16_v_i16m2(&result.Gy[row*W+col], vgy, vl);
            col += vl;
        }
    }

    int   max_l1 = 1;
    float max_l2 = 1.0f;
    for (int i = 0; i < W*H; i++) {
        int mag_l1 = std::abs(result.Gx[i]) + std::abs(result.Gy[i]);
        max_l1 = std::max(max_l1, mag_l1);
        float mag_l2 = std::sqrt((float)result.Gx[i]*result.Gx[i] +
                                  (float)result.Gy[i]*result.Gy[i]);
        max_l2 = std::max(max_l2, mag_l2);
    }

    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int i = row*W+col;
            int gx=result.Gx[i], gy=result.Gy[i];
            int mag_l1=std::abs(gx)+std::abs(gy);
            result.magnitude_l1.data[i]=(uint8_t)(mag_l1*255/max_l1);
            float mag_l2=std::sqrt((float)gx*gx+(float)gy*gy);
            result.magnitude_l2.data[i]=(uint8_t)(mag_l2*255.0f/max_l2);
            int ax=std::abs(gx), ay=std::abs(gy);
            uint8_t dir;
            if      (ay*5<ax*2)  dir=0;
            else if (ay*5<ax*12) dir=45;
            else if (ax*5<ay*2)  dir=90;
            else                 dir=135;
            result.direction.data[i]=dir;
        }
    }
    return result;
}
