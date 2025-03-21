/***************************************************************************
Copyright (c) 2022, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "common.h"

// Optimizes the implementation in ../generic/gemm_ncopy_8.c

int CNAME(BLASLONG m, BLASLONG n, FLOAT *a, BLASLONG lda, FLOAT *b)
{
    BLASLONG i, j;

    FLOAT *a_offset;
    FLOAT *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    FLOAT *b_offset;

    vfloat64m1_t v1, v2, v3, v4, v5, v6, v7, v8;
    vfloat64m1x2_t vx2;
    vfloat64m1x4_t vx4_1, vx4_2;
    vfloat64m1x8_t vx8;
    
    const uint16_t gather_mask[16] = {
        0,4,8,12,
        1,5,9,13,
        2,6,10,14,
        3,7,11,15
    };

    size_t vl;

    //printf("gemm_ncopy_8 m=%ld n=%ld lda=%ld\n", m, n, lda);

    a_offset = a;
    b_offset = b;

    for(j = (n >> 3); j > 0; j--) {
        a_offset1 = a_offset;
        a_offset2 = a_offset + 4*lda;
        a_offset += 8 * lda;

        for(i = m; i >= 8; i -= 8) {
            asm volatile(
                "vsetivli      zero,16,e16,m1,ta,ma;"
                "vle16.v v16, (%[gather_mask]);"

                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vle64.v v0, (%[a_offset1]); addi  t2, %[a_offset1], 32;"
                "vle64.v v4,           (t2); sh3add  t1, %[lda], %[a_offset1];"
                "vle64.v v1,           (t1); sh3add  t2, %[lda], t2;"
                "vle64.v v5,           (t2); sh3add  t1, %[lda], t1;"
                "vle64.v v2,           (t1); sh3add  t2, %[lda], t2;"
                "vle64.v v6,           (t2); sh3add  t1, %[lda], t1;"
                "vle64.v v3,           (t1); sh3add  t2, %[lda], t2;"
                "vle64.v v7,           (t2); addi    %[a_offset1], %[a_offset1], 64;"
                
                "vsetivli        zero,16,e64,m4,ta,ma;"
                "vrgatherei16.vv  v8, v0, v16;"
                "vrgatherei16.vv v12, v4, v16;"
                
                "vsetivli        zero,4,e64,m1,ta,ma;"
                "vse64.v v8, (%[b_offset]); addi  t0, %[b_offset], 64;"
                "vse64.v v9,          (t0); addi  t0, t0, 64;"
                "vse64.v v10,         (t0); addi  t0, t0, 64;"
                "vse64.v v11,         (t0); addi  t0, t0, 64;"
                "vse64.v v12,         (t0); addi  t0, t0, 64;"
                "vse64.v v13,         (t0); addi  t0, t0, 64;"
                "vse64.v v14,         (t0); addi  t0, t0, 64;"
                "vse64.v v15,         (t0); addi  t0, %[b_offset], 32;"
                
                "vle64.v v0, (%[a_offset2]); addi    t2, %[a_offset2], 32;"
                "vle64.v v4,           (t2); sh3add  t1, %[lda], %[a_offset2];"
                "vle64.v v1,           (t1); sh3add  t2, %[lda], t2;"
                "vle64.v v5,           (t2); sh3add  t1, %[lda], t1;"
                "vle64.v v2,           (t1); sh3add  t2, %[lda], t2;"
                "vle64.v v6,           (t2); sh3add  t1, %[lda], t1;"
                "vle64.v v3,           (t1); sh3add  t2, %[lda], t2;"
                "vle64.v v7,           (t2); addi    %[a_offset2], %[a_offset2], 64;"
                
                "vsetivli       zero,16,e64,m4,ta,ma;"
                "vrgatherei16.vv  v8, v0, v16;"
                "vrgatherei16.vv v12, v4, v16;"

                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vse64.v v8,  (t0); addi  t0, t0, 64;"
                "vse64.v v9,  (t0); addi  t0, t0, 64;"
                "vse64.v v10, (t0); addi  t0, t0, 64;"
                "vse64.v v11, (t0); addi  t0, t0, 64;"
                "vse64.v v12, (t0); addi  t0, t0, 64;"
                "vse64.v v13, (t0); addi  t0, t0, 64;"
                "vse64.v v14, (t0); addi  t0, t0, 64;"
                "vse64.v v15, (t0); addi  %[b_offset], %[b_offset], 512;"
                
                : [a_offset1]"+r"(a_offset1), [a_offset2]"+r"(a_offset2), [b_offset]"+r"(b_offset)
                : [lda]"r"(lda), [gather_mask]"r"(gather_mask)
                : "memory",
                "t0","t1","t2","t3","t4","t5","t6",
                "ft0","ft1","ft2","ft3","ft4","ft5","ft6","ft7",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v8","v9","v10","v11","v12","v13","v14","v15","v16"
            );
        }

        if(m & 4) {
            // asm volatile(
            //     "vsetivli      zero,4,e64,m1,ta,ma;"
            //     "vlsseg4e64.v  v0, (%[a_offset1]), %[lda8];"
            //     "vlsseg4e64.v  v4, (%[a_offset2]), %[lda8];"
            //     "addi    %[a_offset1], %[a_offset1], 32;"
            //     "addi    %[a_offset2], %[a_offset2], 32;"
            //     "vse64.v v0, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v4, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v1, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v5, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v2, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v6, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v3, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
            //     "vse64.v v7, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
                
            //     : [a_offset1]"+r"(a_offset1), [a_offset2]"+r"(a_offset2), [b_offset]"+r"(b_offset)
            //     : [lda8]"r"(lda*8)
            //     : "memory",
            //     "t0","t1","t2","t3","t4","t5","t6",
            //     "v0","v1","v2","v3","v4","v5","v6","v7",
            //     "v8","v9","v10","v11","v12","v13","v14","v15"
            // );

            asm volatile(
                "vsetivli      zero,16,e16,m1,ta,ma;"
                "vle16.v v16, (%[gather_mask]);"

                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vle64.v v0, (%[a_offset1]);sh3add  t0, %[lda], %[a_offset1];"
                "vle64.v v1, (t0); sh3add  t1, %[lda], t0;"
                "vle64.v v2, (t1); sh3add  t2, %[lda], t1;"
                "vle64.v v3, (t2); addi    %[a_offset1], %[a_offset1], 32;"
                "vsetivli       zero,16,e64,m4,ta,ma;"
                "vrgatherei16.vv v8, v0, v16;"
                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vse64.v v8, (%[b_offset]); addi  t0, %[b_offset], 64;"
                "vse64.v v9,          (t0); addi  t0, t0, 64;"
                "vse64.v v10,         (t0); addi  t0, t0, 64;"
                "vse64.v v11,         (t0); addi  t0, %[b_offset], 32;"

                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vle64.v v4, (%[a_offset2]); sh3add  t3, %[lda], %[a_offset2];"
                "vle64.v v5,           (t3); sh3add  t4, %[lda], t3;"
                "vle64.v v6,           (t4); sh3add  t5, %[lda], t4;"
                "vle64.v v7,           (t5); addi    %[a_offset2], %[a_offset2], 32;"
                "vsetivli       zero,16,e64,m4,ta,ma;"
                "vrgatherei16.vv v12, v4, v16;"

                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vse64.v v12, (t0); addi  t0, t0, 64;"
                "vse64.v v13, (t0); addi  t0, t0, 64;"
                "vse64.v v14, (t0); addi  t0, t0, 64;"
                "vse64.v v15, (t0); addi    %[b_offset], %[b_offset], 256;"
                
                : [a_offset1]"+r"(a_offset1), [a_offset2]"+r"(a_offset2), [b_offset]"+r"(b_offset)
                : [lda]"r"(lda), [gather_mask]"r"(gather_mask)
                : "memory",
                "t0","t1","t2","t3","t4","t5","t6",
                "ft0","ft1","ft2","ft3","ft4","ft5","ft6","ft7",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v8","v9","v10","v11","v12","v13","v14","v15","v16"
            );
        }
        if(m & 2){
            vfloat64m2x2_t v = __riscv_vlsseg2e64_v_f64m2x2(a_offset1, lda*8, 8);
            __riscv_vse64_v_f64m2(b_offset, __riscv_vget_v_f64m2x2_f64m2(v, 0), 8);
            __riscv_vse64_v_f64m2(b_offset+8, __riscv_vget_v_f64m2x2_f64m2(v, 1), 8);

            a_offset1+=2;
            a_offset2+=2;

            b_offset += 2*8;
        }
        if(m & 1){

            v1 = __riscv_vlse64_v_f64m1(a_offset1, lda*8, 4);
            v2 = __riscv_vlse64_v_f64m1(a_offset2, lda*8, 4);
            __riscv_vse64_v_f64m1(b_offset, v1, 4);
            __riscv_vse64_v_f64m1(b_offset+4, v2, 4);

            a_offset1+=1;
            a_offset2+=1;

            b_offset += 8;
        }
    }

    if (n & 4) {
        a_offset1  = a_offset;
        a_offset += 4 * lda;

        for(i = m; i >= 4; i -= 4) {
            // vx4_1 = __riscv_vlsseg4e64_v_f64m1x4(a_offset1, lda*8, 4);

            // __riscv_vse64_v_f64m4(b_offset, __riscv_vcreate_v_f64m1_f64m4(
            //     __riscv_vget_v_f64m1x4_f64m1(vx4_1, 0),
            //     __riscv_vget_v_f64m1x4_f64m1(vx4_1, 1),
            //     __riscv_vget_v_f64m1x4_f64m1(vx4_1, 2),
            //     __riscv_vget_v_f64m1x4_f64m1(vx4_1, 3)
            // ), 16);

            // a_offset1 += 4;
            // b_offset += 16;
            const uint16_t gather_mask[16] = {
                0,4,8,12,
                1,5,9,13,
                2,6,10,14,
                3,7,11,15
            };

            asm volatile(
                "vsetivli      zero,16,e16,m1,ta,ma;"
                "vle16.v v16, (%[gather_mask]);"

                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vle64.v v0, (%[a_offset1]);sh3add  t0, %[lda], %[a_offset1];"
                "vle64.v v1, (t0); sh3add  t1, %[lda], t0;"
                "vle64.v v2, (t1); sh3add  t2, %[lda], t1;"
                "vle64.v v3, (t2); addi    %[a_offset1], %[a_offset1], 32;"
                "vsetivli       zero,16,e64,m4,ta,ma;"
                "vrgatherei16.vv v4, v0, v16;"
                "vsetivli      zero,4,e64,m1,ta,ma;"
                "vse64.v v4, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
                "vse64.v v5, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
                "vse64.v v6, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
                "vse64.v v7, (%[b_offset]); addi  %[b_offset], %[b_offset], 32;"
                
                : [a_offset1]"+r"(a_offset1), [b_offset]"+r"(b_offset)
                : [lda]"r"(lda), [gather_mask]"r"(gather_mask)
                : "memory",
                "t0","t1","t2","t3","t4","t5","t6",
                "ft0","ft1","ft2","ft3","ft4","ft5","ft6","ft7",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v16"
            );
        }
        if(m & 2){
            vfloat64m1x2_t v = __riscv_vlsseg2e64_v_f64m1x2(a_offset1, lda*8, 4);
            __riscv_vse64_v_f64m1(b_offset, __riscv_vget_v_f64m1x2_f64m1(v, 0), 4);
            __riscv_vse64_v_f64m1(b_offset+4, __riscv_vget_v_f64m1x2_f64m1(v, 1), 4);

            a_offset1+=2;
            a_offset2+=2;

            b_offset += 8;
        }
        if(m & 1){
            v1 = __riscv_vlse64_v_f64m1(a_offset1, lda*8, 4);
            __riscv_vse64_v_f64m1(b_offset, v1, 4);

            a_offset1+=1;

            b_offset += 4;
        }
    }

    if (n & 2) {
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset += 2 * lda;

        for(i = m; i >= 4; i -= 4) {
            vx4_1 = __riscv_vlsseg4e64_v_f64m1x4(a_offset1, lda*8, 2);

            __riscv_vse64_v_f64m1(b_offset, __riscv_vget_v_f64m1x4_f64m1(vx4_1, 0), 2);
            __riscv_vse64_v_f64m1(b_offset+2, __riscv_vget_v_f64m1x4_f64m1(vx4_1, 1), 2);
            __riscv_vse64_v_f64m1(b_offset+4, __riscv_vget_v_f64m1x4_f64m1(vx4_1, 2), 2);
            __riscv_vse64_v_f64m1(b_offset+6, __riscv_vget_v_f64m1x4_f64m1(vx4_1, 3), 2);

            a_offset1 += 4;
            b_offset += 8;
        }
        if(m & 2){
            vfloat64m1x2_t v = __riscv_vlsseg2e64_v_f64m1x2(a_offset1, lda*8, 2);
            __riscv_vse64_v_f64m1(b_offset, __riscv_vget_v_f64m1x2_f64m1(v, 0), 2);
            __riscv_vse64_v_f64m1(b_offset+2, __riscv_vget_v_f64m1x2_f64m1(v, 1), 2);

            a_offset1+=2;
            b_offset += 4;
        }
        
        if(m & 1){
            v1 = __riscv_vlse64_v_f64m1(a_offset1, lda*8, 2);
            __riscv_vse64_v_f64m1(b_offset, v1, 2);

            a_offset1+=1;
            b_offset += 2;
        }
    }

    if (n & 1) {
        a_offset1  = a_offset;

        for(i = m; i >= 4; i -= 4) {
            v1 = __riscv_vle64_v_f64m1(a_offset1, 4);

            __riscv_vse64_v_f64m1(b_offset, v1, 4);

            a_offset1 += 4;
            b_offset += 4;
        }
        if(m & 3){
            vl = __riscv_vsetvl_e64m1(m & 3);
            
            v1 = __riscv_vle64_v_f64m1(a_offset1, vl);

            __riscv_vse64_v_f64m1(b_offset, v1, vl);
        }
    }

    return 0;
}

