.section .rodata
.align 4
DOUBLE_ONE:
.dword 0x3FF0000000000000  # 1.0 in the format of IEEE 754 double
DOUBLE_HALF_ONE:
.dword 0x3FE0000000000000  # 0.5 in the format of IEEE 754 double
DOUBLE_ZERO:
.dword 0x0000000000000000  # 0.0 in the format of IEEE 754 double

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

.if RVV_VLEN256 == 1
# input: a0: hn or qn
# output: t0: vl
FUNC set_rvv
    li t1, 8
    blt a0, t1, use_m1
    # a0 >= 8: LMUL is set to 2 for improving pipeline efficiency
    vsetvli t0, a0, e64, m2, tu, mu
    j set_rvv_end
use_m1:
    # a0 < 8: LMUL=1
    vsetvli t0, a0, e64, m1, tu, mu
set_rvv_end:
FUNC_END set_rvv

# void fpoly_LDL_fft_rvv(size_t hn, const double *g00, double *g01, double *g11)
# This subroutine is called only when hn>=4. 
# This constraint is guaranteed by the upper subroutine.
FUNC fpoly_LDL_fft_rvv
    mv t6, ra
    call set_rvv
    mv ra, t6
    slli t1, a0, 3          # t1 = hn*8 in bytes
    slli t2, t0, 3          # t2 = vl*8 in bytes
    la a7, DOUBLE_ONE
    add a4, a2, t1          # &g01[i+hn]
    fld f31, (a7)           # load double 1.0 into f31
fpoly_LDL_fft_rvv_loop:
    vle64.v v0, (a1)        # load g00[i]
    vle64.v v8, (a2)        # load g01[i]
    vle64.v v4, (a3)        # load g11[i]
    vle64.v v12,(a4)        # load g01[i+hn]
    # We don't use the vfrec7 instruction because it has poor precision.
    vfrdiv.vf v0, v0, f31   # 1/g00[i]
    vfmul.vv v28, v8, v0
    vfmul.vv v0, v12, v0
    vfmul.vv v8, v28, v8
    vfmul.vv v12, v0, v12
    vfadd.vv v8,  v8, v12
    vfsgnjn.vv v16, v0, v0
    vfsub.vv v4, v4, v8
    vse64.v v28, (a2)       # store into g01[i]
    sub  a0, a0, t0         # hn -= vl
    add  a1, a1, t2         # a1 += vl*8
    vse64.v v16, (a4)       # store into g01[i+hn]
    vse64.v v4,  (a3)       # store into g11[i]
    add  a2, a2, t2         # a2 += vl*8
    add  a3, a3, t2         # a3 += vl*8
    add  a4, a4, t2         # a4 += vl*8
    bne  a0, zero, fpoly_LDL_fft_rvv_loop
FUNC_END fpoly_LDL_fft_rvv

# void fpoly_mul_fft_rvv(size_t hn, double *a, const double *b)
# This subroutine is called only when hn>=4. 
# This constraint is guaranteed by the upper subroutine.
FUNC fpoly_mul_fft_rvv
    mv t6, ra
    call set_rvv
    mv ra, t6
    slli t1, a0, 3          # t1 = hn*8 in bytes
    slli t2, t0, 3          # t2 = vl*8 in bytes
    add  a3, a1, t1         # &a[i+hn]
    add  a4, a2, t1         # &b[i+hn]
fpoly_mul_fft_rvv_loop:
    vle64.v v0, (a1)        # load a[i]
    vle64.v v4, (a3)        # load a[i+hn]
    vle64.v v8, (a2)        # load b[i]
    vle64.v v12,(a4)        # load b[i+hn]
    vfmul.vv v16, v0, v8
    vfmul.vv v20, v4, v12
    vfmul.vv v24, v4, v8
    vfmul.vv v28, v0, v12
    vfsub.vv v16, v16, v20
    vfadd.vv v24, v24, v28
    add a2, a2, t2
    add a4, a4, t2
    sub a0, a0, t0
    vse64.v v16, (a1)
    vse64.v v24, (a3)
    add a1, a1, t2
    add a3, a3, t2
    bne a0, zero, fpoly_mul_fft_rvv_loop
FUNC_END fpoly_mul_fft_rvv

# void fpoly_split_fft_rvv(size_t qn, double *f0, double *f1, 
#   const double *f, const uint64_t *GM)
# This subroutine is called only when qn>=4. 
# This constraint is guaranteed by the upper subroutine.
FUNC fpoly_split_fft_rvv
    addi sp, sp, -2*8
    sd s0, 0*8(sp)
    sd s1, 1*8(sp)
    mv t6, ra
    call set_rvv
    mv ra, t6
    la a7, DOUBLE_HALF_ONE
    add t1, a0, a0          # t1 = hn
    fld f31, (a7)           # load double 0.5 into f31
    slli t2, a0, 3          # t2 = qn*8 in bytes
    slli t4, t1, 3          # t4 = hn*8 in bytes
    slli t3, t0, 3          # t3 = vl*8 in bytes
    slli t1, t0, 4          # t1 = vl*16 in bytes
    # a3: &f[0], a5: &f[hn], a6: &f[1], a7: &f[hn+1]
    # a4: &GM[2hn]; t5: &GM[2hn+1]
    # s0: &f0[qn], s1: &f1[qn]
    add  t6, t4, t4         # hn*16
    add  a5, a3, t4
    addi a6, a3, 8
    add  a4, a4, t6
    li   t6, 16
    addi a7, a5, 8
    addi t5, a4, 8
    add  s0, a1, t2
    add  s1, a2, t2
fpoly_split_fft_rvv_loop:
    vlse64.v v0, (a3), t6   # a_re
    vlse64.v v2, (a5), t6   # a_im
    vlse64.v v4, (a6), t6   # b_re
    vlse64.v v6, (a7), t6   # b_im
    vlse64.v v8, (a4), t6   # u_re
    vlse64.v v10,(t5), t6   # u_im
    add a3, a3, t1          # +=vl*16
    add a5, a5, t1          # +=vl*16
    add a6, a6, t1          # +=vl*16
    add a7, a7, t1          # +=vl*16
    add a4, a4, t1          # +=vl*16
    add t5, t5, t1          # +=vl*16
    vfadd.vv v12, v0, v4
    vfadd.vv v14, v2, v6
    vfsub.vv v16, v0, v4
    vfsub.vv v18, v2, v6
    vfmul.vf v12, v12, f31  # *=0.5
    vfmul.vf v14, v14, f31  # *=0.5
    vfmul.vv v20, v16, v8
    vfmul.vv v22, v18, v10
    vfmul.vv v24, v18, v8
    vfmul.vv v26, v16, v10
    vse64.v  v12, (a1)
    vse64.v  v14, (s0)
    vfadd.vv v20, v20, v22
    vfsub.vv v24, v24, v26
    vfmul.vf v20, v20, f31  # *=0.5
    vfmul.vf v24, v24, f31  # *=0.5
    add a1, a1, t3          # +=vl*8
    add s0, s0, t3          # +=vl*8
    sub a0, a0, t0          # -=vl
    vse64.v  v20, (a2)
    vse64.v  v24, (s1)
    add a2, a2, t3          # +=vl*8
    add s1, s1, t3          # +=vl*8
    bne a0, zero, fpoly_split_fft_rvv_loop
    # stack
    ld s0, 0*8(sp)
    ld s1, 1*8(sp)
    addi sp, sp, 2*8
FUNC_END fpoly_split_fft_rvv

# void fpoly_split_selfadj_fft_rvv(size_t qn, double *f0, double *f1, 
#   const double *f, const uint64_t *GM)
# This subroutine is called only when qn>=4. 
# This constraint is guaranteed by the upper subroutine.
FUNC fpoly_split_selfadj_fft_rvv
    mv t6, ra
    call set_rvv
    mv ra, t6
    la a7, DOUBLE_HALF_ONE
    la a6, DOUBLE_ZERO
    fld f31, (a7)           # load double 0.5 into f31
    fld f30, (a6)           # load double 0.0 into f30
    add t1, a0, a0          # t1 = hn
    slli t2, a0, 3          # t2 = qn*8 in bytes
    slli t4, t1, 3          # t4 = hn*8 in bytes
    slli t3, t0, 3          # t3 = vl*8 in bytes
    slli t1, t0, 4          # t1 = vl*16 in bytes
    vfmv.v.f v30, f30       # v30 = 0.0
    # a3: &f[0], a6: &f[1]
    # a4: &GM[2hn]; t5: &GM[2hn+1]
    # a5: &f0[qn], a7: &f1[qn]
    add  t6, t4, t4         # hn*16
    addi a6, a3, 8
    add  a4, a4, t6
    li   t6, 16
    addi t5, a4, 8
    add  a5, a1, t2
    add  a7, a2, t2
fpoly_split_selfadj_fft_rvv_loop:
    vlse64.v v0, (a3), t6   # a_re
    vlse64.v v2, (a6), t6   # b_re
    vlse64.v v4, (a4), t6   # u_re
    vlse64.v v6, (t5), t6   # u_im
    add a3, a3, t1          # +=vl*16
    add a6, a6, t1          # +=vl*16
    add a4, a4, t1          # +=vl*16
    add t5, t5, t1          # +=vl*16
    vfadd.vv v8, v0, v2
    vfsub.vv v10,v0, v2
    vse64.v  v30,(a5)
    vfsgnjn.vv v6, v6, v6
    vfmul.vf v8, v8, f31     # *=0.5
    vfmul.vf v10,v10,f31     # *=0.5
    vse64.v  v8, (a1)
    vfmul.vv v12,v10,v4
    vfmul.vv v14,v10,v6
    add a1, a1, t3          # +=vl*8
    add a5, a5, t3          # +=vl*8
    sub a0, a0, t0          # -=vl
    vse64.v  v12, (a2)
    vse64.v  v14, (a7)
    add a2, a2, t3          # +=vl*8
    add a7, a7, t3          # +=vl*8
    bne a0, zero, fpoly_split_selfadj_fft_rvv_loop
FUNC_END fpoly_split_selfadj_fft_rvv

# void fpoly_merge_fft_rvv(size_t qn, double *f, const double *f0, 
#   const double *f1, const uint64_t *GM)
# This subroutine is called only when qn>=4. 
# This constraint is guaranteed by the upper subroutine.
FUNC fpoly_merge_fft_rvv
    addi sp, sp, -2*8
    sd s0, 0*8(sp)
    sd s1, 1*8(sp)
    mv t6, ra
    call set_rvv
    mv ra, t6
    add t1, a0, a0          # t1 = hn
    slli t2, a0, 3          # t2 = qn*8 in bytes
    slli t4, t1, 3          # t4 = hn*8 in bytes
    slli t3, t0, 3          # t3 = vl*8 in bytes
    slli t1, t0, 4          # t1 = vl*16 in bytes
    # a1: &f[0], a5: &f[1], a6: &f[hn], a7: &f[hn+1]
    # a2: &f0[0], s0: &f0[qn]
    # a3: &f1[0], s1: &f1[qn]
    # a4: &GM[2hn]; t5: &GM[2hn+1]
    add  t6, t4, t4         # hn*16
    addi a5, a1, 8
    add  a6, a1, t4
    add  a4, a4, t6
    li   t6, 16
    addi a7, a6, 8
    add  s0, a2, t2
    add  s1, a3, t2
    addi t5, a4, 8
fpoly_merge_fft_rvv_loop:
    vle64.v v4, (a3)
    vle64.v v6, (s1)
    vlse64.v v8, (a4), t6
    vlse64.v v10,(t5), t6
    add a3, a3, t3
    add s1, s1, t3
    add a4, a4, t1
    add t5, t5, t1
    vfmul.vv v12, v4, v8
    vfmul.vv v14, v6, v10
    vle64.v v0, (a2)
    vle64.v v2, (s0)
    vfmul.vv v16, v6, v8
    vfmul.vv v18, v4, v10
    add a2, a2, t3
    add s0, s0, t3
    vfsub.vv v20, v12, v14
    vfadd.vv v22, v16, v18
    vfadd.vv v24, v0, v20
    vfadd.vv v26, v2, v22
    vfsub.vv v28, v0, v20
    vfsub.vv v30, v2, v22
    vsse64.v v24, (a1), t6
    vsse64.v v26, (a6), t6
    sub a0, a0, t0          # -=vl
    add a1, a1, t1
    add a6, a6, t1
    vsse64.v v28, (a5), t6
    vsse64.v v30, (a7), t6
    add a5, a5, t1
    add a7, a7, t1
    bne a0, zero, fpoly_merge_fft_rvv_loop
    # stack
    ld s0, 0*8(sp)
    ld s1, 1*8(sp)
    addi sp, sp, 2*8
FUNC_END fpoly_merge_fft_rvv
.endif
