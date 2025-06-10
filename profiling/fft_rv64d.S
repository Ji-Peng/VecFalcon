#include "fft_rv64d_consts.h"

.section .rodata
.align 4
ONE_OVER_2_POW_9:
.dword 4571153621781053440 # 1/2^9
ONE_OVER_2_POW_10:
.dword 4566650022153682944 # 1/2^10

.text

.macro FUNC name
    .global \name
    .align 2
    \name:
.endm

.macro FUNC_END name
    ret
    .type \name, %function
    .size \name, .-\name
.endm

.macro save_regs
  sd s0,  0*8(sp)
  sd s1,  1*8(sp)
  sd s2,  2*8(sp)
  sd s3,  3*8(sp)
  sd s4,  4*8(sp)
  sd s5,  5*8(sp)
  sd s6,  6*8(sp)
  sd s7,  7*8(sp)
.endm

.macro restore_regs
  ld s0,  0*8(sp)
  ld s1,  1*8(sp)
  ld s2,  2*8(sp)
  ld s3,  3*8(sp)
  ld s4,  4*8(sp)
  ld s5,  5*8(sp)
  ld s6,  6*8(sp)
  ld s7,  7*8(sp)
.endm

.macro CT_BFx2 a0re,b0re,a1re,b1re,a0im,b0im,a1im,b1im,\
		w0re,w1re,w0im,w1im,tf00,tf01,tf10,tf11
	fmul.d \tf00, \b0re, \w0re
	fmul.d \tf01, \b0im, \w0im
	fmul.d \tf10, \b1re, \w1re
	fmul.d \tf11, \b1im, \w1im
	fmul.d \b0re, \b0re, \w0im
	fmul.d \b0im, \b0im, \w0re
	fmul.d \b1re, \b1re, \w1im
	fmul.d \b1im, \b1im, \w1re
	fsub.d \tf00, \tf00, \tf01
	fsub.d \tf10, \tf10, \tf11
	fadd.d \tf01, \b0re, \b0im
	fadd.d \tf11, \b1re, \b1im
	fsub.d \b0re, \a0re, \tf00
	fsub.d \b1re, \a1re, \tf10
	fsub.d \b0im, \a0im, \tf01
	fsub.d \b1im, \a1im, \tf11
	fadd.d \a0re, \a0re, \tf00
	fadd.d \a1re, \a1re, \tf10
	fadd.d \a0im, \a0im, \tf01
	fadd.d \a1im, \a1im, \tf11
.endm

.macro GS_BFx2 a0re,b0re,a1re,b1re,a0im,b0im,a1im,b1im,\
        w0re,w1re,w0im,w1im,tf00,tf01,tf10,tf11
    fsub.d \tf00, \a0re, \b0re
    fsub.d \tf01, \a0im, \b0im
    fsub.d \tf10, \a1re, \b1re
    fsub.d \tf11, \a1im, \b1im
    fadd.d \a0re, \a0re, \b0re
    fadd.d \a0im, \a0im, \b0im
    fadd.d \a1re, \a1re, \b1re
    fadd.d \a1im, \a1im, \b1im
    fmul.d \b0re, \tf00, \w0re // add op 1 - 0th
    fmul.d \b0im, \tf01, \w0re // sub op 1 - 0th
    fmul.d \b1re, \tf10, \w1re // add op 1 - 1th
    fmul.d \b1im, \tf11, \w1re // sub op 1 - 1th
    fmul.d \tf01, \tf01, \w0im // add op 2 - 0th
    fmul.d \tf00, \tf00, \w0im // sub op 2 - 0th
    fmul.d \tf11, \tf11, \w1im // add op 2 - 1th
    fmul.d \tf10, \tf10, \w1im // sub op 2 - 1th
    fadd.d \b0re, \b0re, \tf01
    fsub.d \b0im, \b0im, \tf00
    fadd.d \b1re, \b1re, \tf11
    fsub.d \b1im, \b1im, \tf10
.endm

# void fpoly_FFT_rv64d(unsigned logn, double *f, double *GM);
# t0: n, t1: hn, t2: byte_stride=ht*8, t3: loop number
### Register usage:
# f16-f31: 8 complex numbers
# f0-f11: items of GM table
# a6,a7,t5,t6: items of GM table
# a0,a1,a2: parameters
# a3,a4,a5,s0-s7: addresses related to input double *f
# t0, t2, t4: temporary usage
# t1: hn; t3: loop number
FUNC fpoly_FFT_rv64d
    li t6, 1
    addi sp, sp, -8*9
    sll  t0, t6, a0 # n = 1 << logn
    save_regs
    srli t1, t0, 1  # hn = n >> 1
    srli t2, t0, 1  # byte_stride=n>>1=ht*8, ht=64 for Falcon1024, =32 for Falcon512
    srli t3, t1, 3  # loop number=hn/8=hn >> 3
    ### prepare the base addresses required by loading
    # a1: &f[0*ht], a3: &f[1*ht], a4: &f[2*ht], a5: &f[3*ht]
    # s0: &f[4*ht], s1: &f[5*ht], s2: &f[6*ht], s3: &f[7*ht]
    add  t4, t2, t2;    add  a3, a1, t2
    add  a4, a1, t4;    add  a5, a3, t4
    add  s0, a4, t4;    add  s1, a5, t4
    add  s2, s0, t4;    add  s3, s1, t4
    sd a1, 8*8(sp)  # save a1 for later use
    ### GM:
    # f0|f1, f2-f3|f4-f5, and f6-f9|f10-f11 for layer 1, 2, and 3 respectively.
    fld f0, 0*8(a2);  fld f1, 1*8(a2)
    fld f2, 2*8(a2);  fld f3, 3*8(a2)
    fld f4, 4*8(a2);  fld f5, 5*8(a2)
    fld f6, 6*8(a2);  fld f7, 7*8(a2)
    fld f8, 8*8(a2);  fld f9, 9*8(a2)
    ld a6, 10*8(a2);  ld a7, 11*8(a2)
    ld t5, 12*8(a2);  ld t6, 13*8(a2)
    addi a2, a2, 14*8 # jump to the beginning of 4th layer
### 1 2 3 layers merging
fpoly_FFT_rv64d_123_loop:
    ### load coefficients
    # f16-f23=f[0*ht,1*ht,2*ht,3*ht,4*ht,5*ht,6*ht,7*ht]
    # f24-f31 are imaginary parts
    slli t4, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f20, (s0);  fld f21, (s1)
    add s4, a1, t4; add s5, a3, t4
    fld f24, (s4);  fld f25, (s5)
    add s4, s0, t4; add s5, s1, t4
    fld f28, (s4);  fld f29, (s5)
    fld f18, (a4);  fld f19, (a5)
    fld f22, (s2);  fld f23, (s3)
    add s6, a4, t4; add s7, a5, t4
    fld f26, (s6);  fld f27, (s7)
    add s6, s2, t4; add s7, s3, t4
    fld f30, (s6);  fld f31, (s7)
    ### layer 1
    CT_BFx2 f16,f20,f17,f21,f24,f28,f25,f29,f0,f0,f1,f1,f12,f13,f14,f15
    CT_BFx2 f18,f22,f19,f23,f26,f30,f27,f31,f0,f0,f1,f1,f12,f13,f14,f15
    ### layer 2
    CT_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f2,f2,f4,f4,f12,f13,f14,f15
    fmv.d.x f10, a6; fmv.d.x f11, a7
    CT_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f3,f3,f5,f5,f12,f13,f14,f15
    ### layer 3
    CT_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f6,f7,f10,f11,f12,f13,f14,f15
    fmv.d.x f10, t5; fmv.d.x f11, t6
    CT_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f8,f9,f10,f11,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    add s4, a1, t4; add s5, a3, t4
    fsd f24, (s4);  fsd f25, (s5)
    add s6, a4, t4; add s7, a5, t4
    fsd f26, (s6);  fsd f27, (s7)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, s0, t4; add s5, s1, t4
    fsd f28, (s4);  fsd f29, (s5)
    add s6, s2, t4; add s7, s3, t4
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8; addi a3, a3, 8
    addi a4, a4, 8; addi a5, a5, 8
    addi s0, s0, 8; addi s1, s1, 8
    addi t3, t3, -1
    addi s2, s2, 8; addi s3, s3, 8
    bnez t3, fpoly_FFT_rv64d_123_loop
    ld a1, 8*8(sp)  # restore a1
    ### prepare the base addresses required by loading; ht=n>>7; 8ht=n>>4
    # a1: &f[0*ht], a3: &f[1*ht], a4: &f[2*ht], a5: &f[3*ht]
    # s0: &f[4*ht], s1: &f[5*ht], s2: &f[6*ht], s3: &f[7*ht]
    srli t2, t0, 4
    srli t4, t0, 3;     add  a3, a1, t2
    add  a4, a1, t4;    add  a5, a3, t4
    add  s0, a4, t4;    add  s1, a5, t4
    add  s2, s0, t4;    add  s3, s1, t4
    # t3 for controlling outer loop
    li t3, 0
fpoly_FFT_rv64d_456_outer_loop:
    # t4 for controlling inner loop
    li t4, 0
    ### GM:
    # f0|f1, f2-f3|f4-f5, and f6-f9|f10-f11 for layer 4, 5, and 6 respectively.
    fld f0, 0*8(a2);  fld f1, 1*8(a2)
    fld f2, 2*8(a2);  fld f3, 3*8(a2)
    fld f4, 4*8(a2);  fld f5, 5*8(a2)
    fld f6, 6*8(a2);  fld f7, 7*8(a2)
    fld f8, 8*8(a2);  fld f9, 9*8(a2)
    ld a6, 10*8(a2);  ld a7, 11*8(a2)
    ld t5, 12*8(a2);  ld t6, 13*8(a2)
    addi a2, a2, 14*8 # jump to the beginning of 4th layer of next iteration
fpoly_FFT_rv64d_456_inner_loop:
    ### load coefficients
    # f16-f23=f[0*ht,1*ht,2*ht,3*ht,4*ht,5*ht,6*ht,7*ht]
    # f24-f31 are imaginary parts
    slli t2, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 4
    CT_BFx2 f16,f20,f17,f21,f24,f28,f25,f29,f0,f0,f1,f1,f12,f13,f14,f15
    CT_BFx2 f18,f22,f19,f23,f26,f30,f27,f31,f0,f0,f1,f1,f12,f13,f14,f15
    ### layer 5
    CT_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f2,f2,f4,f4,f12,f13,f14,f15
    fmv.d.x f10, a6; fmv.d.x f11, a7
    CT_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f3,f3,f5,f5,f12,f13,f14,f15
    ### layer 6
    CT_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f6,f7,f10,f11,f12,f13,f14,f15
    fmv.d.x f10, t5; fmv.d.x f11, t6
    CT_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f8,f9,f10,f11,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8; addi a3, a3, 8
    addi a4, a4, 8; addi a5, a5, 8
    addi s0, s0, 8; addi s1, s1, 8
    srli s6, t0, 7 # ht=4 for Falcon512, ht=8 for Falcon1024
    addi t4, t4, 1
    addi s2, s2, 8; addi s3, s3, 8
    blt t4, s6, fpoly_FFT_rv64d_456_inner_loop
fpoly_FFT_rv64d_456_inner_loop_end:
    # 8*8ht and 1*8ht
    slli s4, s6, 6
    slli s5, s6, 3
    li s7, 8
    addi t3, t3, 1
    sub s4, s4, s5
    add a1, a1, s4; add a3, a3, s4
    add a4, a4, s4; add a5, a5, s4
    add s0, s0, s4; add s1, s1, s4
    add s2, s2, s4; add s3, s3, s4
    blt t3, s7, fpoly_FFT_rv64d_456_outer_loop
    ld a1, 8*8(sp)  # restore a1
    ### prepare the base addresses required by loading; ht=1
    # a1: &f[0*ht], a3: &f[1*ht], a4: &f[2*ht], a5: &f[3*ht]
    # s0: &f[4*ht], s1: &f[5*ht], s2: &f[6*ht], s3: &f[7*ht]
    addi a3, a1, 8
    addi a4, a1, 16;    addi a5, a3, 16
    addi s0, a4, 16;    addi s1, a5, 16
    addi s2, s0, 16;    addi s3, s1, 16
    # loop number=hn/8
    li s4, 256
    srli t3, t1, 3
    bne t1, s4, fpoly_FFT_rv64d_789_loop
fpoly_FFT_rv64d_78_loop:
    ### GM:
    # f0-f1|f2-f3, f4-f7|f8-f11 for layer 7 and 8 respectively.
    fld f0, 0*8(a2);  fld f1, 1*8(a2)
    fld f2, 2*8(a2);  fld f3, 3*8(a2)
    ### load coefficients
    slli t2, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 7
    CT_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f0,f0,f2,f2,f12,f13,f14,f15
    fld f4, 4*8(a2);  fld f5, 5*8(a2)
    fld f8, 8*8(a2);  fld f9, 9*8(a2)
    CT_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f1,f1,f3,f3,f12,f13,f14,f15
    fld f6, 6*8(a2);  fld f7, 7*8(a2)
    fld f10,10*8(a2); fld f11,11*8(a2)
    addi a2, a2, 12*8 # jump to the beginning of 7th layer of next iteration
    ### layer 8
    CT_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f4,f5,f8, f9, f12,f13,f14,f15
    CT_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f6,f7,f10,f11,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8*8; addi a3, a3, 8*8
    addi a4, a4, 8*8; addi a5, a5, 8*8
    addi s0, s0, 8*8; addi s1, s1, 8*8
    addi t3, t3, -1
    addi s2, s2, 8*8; addi s3, s3, 8*8
    bnez t3, fpoly_FFT_rv64d_78_loop
    j fpoly_FFT_rv64d_end
fpoly_FFT_rv64d_789_loop:
    ### GM:
    # f0|f1, f2-f3|f4-f5, and f4-f7|f8-f11 for layer 7, 8, and 9 respectively.
    fld f0, 0*8(a2);  fld f1, 1*8(a2)
    ### load coefficients
    slli t2, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 7
    CT_BFx2 f16,f20,f17,f21,f24,f28,f25,f29,f0,f0,f1,f1,f12,f13,f14,f15
    fld f2, 2*8(a2);  fld f3, 3*8(a2)
    fld f4, 4*8(a2);  fld f5, 5*8(a2)
    CT_BFx2 f18,f22,f19,f23,f26,f30,f27,f31,f0,f0,f1,f1,f12,f13,f14,f15
    ### layer 8
    CT_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f2,f2,f4,f4,f12,f13,f14,f15
    fld f6, 8*8(a2);  fld f7, 9*8(a2)
    fld f8, 10*8(a2); fld f9, 11*8(a2)
    CT_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f3,f3,f5,f5,f12,f13,f14,f15
    ### layer 9
    fld f4, 6*8(a2);  fld f5, 7*8(a2)
    fld f10,12*8(a2); fld f11,13*8(a2)
    addi a2, a2, 14*8 # jump to the beginning of 7th layer of next iteration
    CT_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f4,f5, f8, f9,f12,f13,f14,f15
    CT_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f6,f7,f10,f11,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8*8; addi a3, a3, 8*8
    addi a4, a4, 8*8; addi a5, a5, 8*8
    addi s0, s0, 8*8; addi s1, s1, 8*8
    addi t3, t3, -1
    addi s2, s2, 8*8; addi s3, s3, 8*8
    bnez t3, fpoly_FFT_rv64d_789_loop
fpoly_FFT_rv64d_end:
    restore_regs
    addi sp, sp, 8*9
FUNC_END fpoly_FFT_rv64d

# void fpoly_iFFT_rv64d(unsigned logn, double *f, double *GM);
# t0: n, t1: hn, t2: byte_stride=ht*8, t3: loop number
### Register usage:
# f16-f31: 8 complex numbers
# f0-f11: items of GM table
# a6,a7,t5,t6: items of GM table
# a0,a1,a2: parameters
# a3,a4,a5,s0-s7: addresses related to input double *f
# t0, t2, t4: temporary usage
# t1: hn; t3: loop number
FUNC fpoly_iFFT_rv64d
    li t6, 1
    addi sp, sp, -8*10
    sll  t0, t6, a0 # n = 1 << logn
    save_regs
    srli t1, t0, 1  # hn = n >> 1
    srli t3, t1, 3  # loop number=hn/8=hn >> 3
    sd a1, 8*8(sp)  # store a1
    sd a2, 9*8(sp)  # store a2
    ### prepare the base addresses required by loading; ht=1
    # a1: &f[0*ht], a3: &f[1*ht], a4: &f[2*ht], a5: &f[3*ht]
    # s0: &f[4*ht], s1: &f[5*ht], s2: &f[6*ht], s3: &f[7*ht]
    addi a3, a1, 8
    addi a4, a1, 16;    addi a5, a3, 16
    addi s0, a4, 16;    addi s1, a5, 16
    li s4, 256
    addi s2, s0, 16;    addi s3, s1, 16
    bne t1, s4, fpoly_iFFT_rv64d_987_start
fpoly_iFFT_rv64d_87_start:
    # t2, t4: start GM index of layer 8 and 7
    addi t2, a2, GM332_l8_IDX*8
    addi t4, a2, GM332_l7_IDX*8
fpoly_iFFT_rv64d_87_loop:
    ### GM:
    # f4-f7|f8-f11, f0-f1|f2-f3 for layer 8 and 7 respectively.
    fld f4, 0*8(t2);  fld f5, 1*8(t2)
    fld f8, 4*8(t2);  fld f9, 5*8(t2)
    fld f6, 2*8(t2);  fld f7, 3*8(t2)
    fld f10,6*8(t2);  fld f11,7*8(t2)
    ### load coefficients
    slli a6, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, a6; add s5, a3, a6
    add s6, a4, a6; add s7, a5, a6
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, a6; add s5, s1, a6
    add s6, s2, a6; add s7, s3, a6
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 8
    GS_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f4,f5, f8, f9,f12,f13,f14,f15
    fld f0, 0*8(t4);  fld f1, 1*8(t4)
    fld f2, 2*8(t4);  fld f3, 3*8(t4)
    # jump to the next iteration
    addi t2, t2, GM332_l7_8_NUM*8;  addi t4, t4, GM332_l7_8_NUM*8
    GS_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f6,f7,f10,f11,f12,f13,f14,f15
    ### layer 7
    GS_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f0,f0,f2,f2,f12,f13,f14,f15
    GS_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f1,f1,f3,f3,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, a6; add s5, a3, a6
    add s6, a4, a6; add s7, a5, a6
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, a6; add s5, s1, a6
    add s6, s2, a6; add s7, s3, a6
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8*8; addi a3, a3, 8*8
    addi a4, a4, 8*8; addi a5, a5, 8*8
    addi s0, s0, 8*8; addi s1, s1, 8*8
    addi t3, t3, -1
    addi s2, s2, 8*8; addi s3, s3, 8*8
    bnez t3, fpoly_iFFT_rv64d_87_loop
    j fpoly_iFFT_rv64d_987_loop_end
fpoly_iFFT_rv64d_987_start:
    # t2, t4, t5: start GM index of layer 9, 8, and 7
    addi t2, a2, GM333_l9_IDX*8
    addi t4, a2, GM333_l8_IDX*8
    addi t5, a2, GM333_l7_IDX*8
fpoly_iFFT_rv64d_987_loop:
    ### GM:
    # f4-f7|f8-f11, f2-f3|f4-f5, and f0|f1 for layer 9, 8, and 7 respectively.
    ### load coefficients
    slli a6, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, a6; add s5, a3, a6
    add s6, a4, a6; add s7, a5, a6
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, a6; add s5, s1, a6
    add s6, s2, a6; add s7, s3, a6
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 9
    fld f4, 0*8(t2);  fld f5, 1*8(t2)
    fld f8, 4*8(t2);  fld f9, 5*8(t2)
    fld f6, 2*8(t2);  fld f7, 3*8(t2)
    fld f10,6*8(t2);  fld f11,7*8(t2)
    addi t2, t2, GM333_l7_9_NUM*8
    GS_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f4,f5, f8, f9,f12,f13,f14,f15
    fld f2, 0*8(t4);  fld f4, 2*8(t4)
    GS_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f6,f7,f10,f11,f12,f13,f14,f15
    ### layer 8
    fld f3, 1*8(t4);  fld f5, 3*8(t4)
    addi t4, t4, GM333_l7_9_NUM*8
    GS_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f2,f2,f4,f4,f12,f13,f14,f15
    fld f0, 0*8(t5);  fld f1, 1*8(t5)
    addi t5, t5, GM333_l7_9_NUM*8
    GS_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f3,f3,f5,f5,f12,f13,f14,f15
    ### layer 7
    GS_BFx2 f16,f20,f17,f21,f24,f28,f25,f29,f0,f0,f1,f1,f12,f13,f14,f15
    GS_BFx2 f18,f22,f19,f23,f26,f30,f27,f31,f0,f0,f1,f1,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, a6; add s5, a3, a6
    add s6, a4, a6; add s7, a5, a6
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, a6; add s5, s1, a6
    add s6, s2, a6; add s7, s3, a6
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8*8; addi a3, a3, 8*8
    addi a4, a4, 8*8; addi a5, a5, 8*8
    addi s0, s0, 8*8; addi s1, s1, 8*8
    addi t3, t3, -1
    addi s2, s2, 8*8; addi s3, s3, 8*8
    bnez t3, fpoly_iFFT_rv64d_987_loop
fpoly_iFFT_rv64d_987_loop_end:
    ld a1, 8*8(sp)  # restore a1
    ### prepare the base addresses required by loading; ht=n>>7; 8ht=n>>4
    # a1: &f[0*ht], a3: &f[1*ht], a4: &f[2*ht], a5: &f[3*ht]
    # s0: &f[4*ht], s1: &f[5*ht], s2: &f[6*ht], s3: &f[7*ht]
    srli t2, t0, 4
    srli t4, t0, 3;     add  a3, a1, t2
    add  a4, a1, t4;    add  a5, a3, t4
    add  s0, a4, t4;    add  s1, a5, t4
    add  s2, s0, t4;    add  s3, s1, t4
    ### prepare GM address
    addi a2, a2, GM332_l4_IDX*8
    # t3 for controlling outer loop
    li t3, 0
fpoly_iFFT_rv64d_654_outer_loop:
    # t4 for controlling inner loop
    li t4, 0
    ### GM:
    # f0|f1, f2-f3|f4-f5, and f6-f9|f10-f11 for layer 4, 5, and 6 respectively.
    fld f0, 0*8(a2);  fld f1, 1*8(a2)
    fld f2, 2*8(a2);  fld f3, 3*8(a2)
    fld f4, 4*8(a2);  fld f5, 5*8(a2)
    fld f6, 6*8(a2);  fld f7, 7*8(a2)
    fld f8, 8*8(a2);  fld f9, 9*8(a2)
    ld a6, 10*8(a2);  ld a7, 11*8(a2)
    ld t5, 12*8(a2);  ld t6, 13*8(a2)
    addi a2, a2, GM332_l4_6_NUM*8 # jump to the beginning of 4th layer of next iteration
fpoly_iFFT_rv64d_654_inner_loop:
    ### load coefficients
    # f16-f23=f[0*ht,1*ht,2*ht,3*ht,4*ht,5*ht,6*ht,7*ht]
    # f24-f31 are imaginary parts
    slli t2, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 6
    fmv.d.x f10, a6; fmv.d.x f11, a7
    GS_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f6,f7,f10,f11,f12,f13,f14,f15
    fmv.d.x f10, t5; fmv.d.x f11, t6
    GS_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f8,f9,f10,f11,f12,f13,f14,f15
    ### layer 5
    GS_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f2,f2,f4,f4,f12,f13,f14,f15
    GS_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f3,f3,f5,f5,f12,f13,f14,f15
    ### layer 4
    GS_BFx2 f16,f20,f17,f21,f24,f28,f25,f29,f0,f0,f1,f1,f12,f13,f14,f15
    GS_BFx2 f18,f22,f19,f23,f26,f30,f27,f31,f0,f0,f1,f1,f12,f13,f14,f15
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, t2; add s5, a3, t2
    add s6, a4, t2; add s7, a5, t2
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, t2; add s5, s1, t2
    add s6, s2, t2; add s7, s3, t2
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8; addi a3, a3, 8
    addi a4, a4, 8; addi a5, a5, 8
    addi s0, s0, 8; addi s1, s1, 8
    srli s6, t0, 7 # ht=4 for Falcon512, ht=8 for Falcon1024
    addi t4, t4, 1
    addi s2, s2, 8; addi s3, s3, 8
    blt t4, s6, fpoly_iFFT_rv64d_654_inner_loop
fpoly_iFFT_rv64d_654_inner_loop_end:
    # 8*8ht and 1*8ht
    slli s4, s6, 6
    slli s5, s6, 3
    li s7, 8
    sub s4, s4, s5
    addi t3, t3, 1
    add a1, a1, s4; add a3, a3, s4
    add a4, a4, s4; add a5, a5, s4
    add s0, s0, s4; add s1, s1, s4
    add s2, s2, s4; add s3, s3, s4
    blt t3, s7, fpoly_iFFT_rv64d_654_outer_loop
    ld a2, 9*8(sp)  # restore a2
    ld a1, 8*8(sp)  # restore a1
    srli t2, t0, 1  # byte_stride=n>>1=ht*8, ht=64 for Falcon1024, =32 for Falcon512
    srli t3, t1, 3  # loop number=hn/8=hn >> 3
    ### GM:
    # f0|f1, f2-f3|f4-f5, and f6-f9|f10-f11 for layer 1, 2, and 3 respectively.
    fld f0, 0*8(a2);  fld f1, 1*8(a2)
    fld f2, 2*8(a2);  fld f3, 3*8(a2)
    fld f4, 4*8(a2);  fld f5, 5*8(a2)
    fld f6, 6*8(a2);  fld f7, 7*8(a2)
    fld f8, 8*8(a2);  fld f9, 9*8(a2)
    ld a6, 10*8(a2);  ld a7, 11*8(a2)
    ld t5, 12*8(a2);  ld t6, 13*8(a2)
    ### prepare the base addresses required by loading
    # a1: &f[0*ht], a3: &f[1*ht], a4: &f[2*ht], a5: &f[3*ht]
    # s0: &f[4*ht], s1: &f[5*ht], s2: &f[6*ht], s3: &f[7*ht]
    add  t4, t2, t2;    add  a3, a1, t2
    add  a4, a1, t4;    add  a5, a3, t4
    add  s0, a4, t4;    add  s1, a5, t4
    add  s2, s0, t4;    add  s3, s1, t4
    # load 1/2^n
    li t0, 256
    la t2, ONE_OVER_2_POW_9
    bne t1, t0, load_ONE_OVER_2_POW_10
load_ONE_OVER_2_POW_9:
    ld a0, 0*8(t2)
    j fpoly_iFFT_rv64d_321_loop
load_ONE_OVER_2_POW_10:
    ld a0, 1*8(t2)
fpoly_iFFT_rv64d_321_loop:
    ### load coefficients
    # f16-f23=f[0*ht,1*ht,2*ht,3*ht,4*ht,5*ht,6*ht,7*ht]
    # f24-f31 are imaginary parts
    slli t4, t1, 3
    fld f16, (a1);  fld f17, (a3)
    fld f18, (a4);  fld f19, (a5)
    fld f20, (s0);  fld f21, (s1)
    fld f22, (s2);  fld f23, (s3)
    add s4, a1, t4; add s5, a3, t4
    add s6, a4, t4; add s7, a5, t4
    fld f24, (s4);  fld f25, (s5)
    fld f26, (s6);  fld f27, (s7)
    add s4, s0, t4; add s5, s1, t4
    add s6, s2, t4; add s7, s3, t4
    fld f28, (s4);  fld f29, (s5)
    fld f30, (s6);  fld f31, (s7)
    ### layer 3
    fmv.d.x f10, a6; fmv.d.x f11, a7
    GS_BFx2 f16,f17,f18,f19,f24,f25,f26,f27,f6,f7,f10,f11,f12,f13,f14,f15
    fmv.d.x f10, t5; fmv.d.x f11, t6
    GS_BFx2 f20,f21,f22,f23,f28,f29,f30,f31,f8,f9,f10,f11,f12,f13,f14,f15
    ### layer 2
    GS_BFx2 f16,f18,f17,f19,f24,f26,f25,f27,f2,f2,f4,f4,f12,f13,f14,f15
    GS_BFx2 f20,f22,f21,f23,f28,f30,f29,f31,f3,f3,f5,f5,f12,f13,f14,f15
    ### layer 1
    fmv.d.x f10, a0
    GS_BFx2 f16,f20,f17,f21,f24,f28,f25,f29,f0,f0,f1,f1,f12,f13,f14,f15
    fmul.d f16, f16, f10; fmul.d f17, f17, f10
    fmul.d f24, f24, f10; fmul.d f25, f25, f10
    fmul.d f20, f20, f10; fmul.d f21, f21, f10
    fmul.d f28, f28, f10; fmul.d f29, f29, f10
    GS_BFx2 f18,f22,f19,f23,f26,f30,f27,f31,f0,f0,f1,f1,f12,f13,f14,f15
    fmul.d f18, f18, f10; fmul.d f19, f19, f10
    fmul.d f26, f26, f10; fmul.d f27, f27, f10
    fmul.d f22, f22, f10; fmul.d f23, f23, f10
    fmul.d f30, f30, f10; fmul.d f31, f31, f10
    ### store results
    fsd f16, (a1);  fsd f17, (a3)
    fsd f18, (a4);  fsd f19, (a5)
    fsd f20, (s0);  fsd f21, (s1)
    fsd f22, (s2);  fsd f23, (s3)
    add s4, a1, t4; add s5, a3, t4
    add s6, a4, t4; add s7, a5, t4
    fsd f24, (s4);  fsd f25, (s5)
    fsd f26, (s6);  fsd f27, (s7)
    add s4, s0, t4; add s5, s1, t4
    add s6, s2, t4; add s7, s3, t4
    fsd f28, (s4);  fsd f29, (s5)
    fsd f30, (s6);  fsd f31, (s7)
    addi a1, a1, 8; addi a3, a3, 8
    addi a4, a4, 8; addi a5, a5, 8
    addi s0, s0, 8; addi s1, s1, 8
    addi t3, t3, -1
    addi s2, s2, 8; addi s3, s3, 8
    bnez t3, fpoly_iFFT_rv64d_321_loop
    restore_regs
    addi sp, sp, 8*10
FUNC_END fpoly_iFFT_rv64d
