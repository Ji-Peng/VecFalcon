.section .rodata
.align 4
_MASK_0101:
.dword 0,1,0,1
_MASK_2323:
.dword 2,3,2,3
_MASK_0022:
.dword 0,0,2,2
_MASK_1133:
.dword 1,1,3,3
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

// shuffle2
# f[0ht, 1ht, 2ht, 3ht],f[4ht, 5ht, 6ht, 7ht] -> 
# f[0ht, 1ht, 4ht, 5ht],f[2ht, 3ht, 6ht, 7ht]
# vm0/vm1: _MASK_0101/_MASK_2323
# v0: _MASK_V0_0b0011
// shuffle1
# f[0ht, 1ht, 4ht, 5ht],f[2ht, 3ht, 6ht, 7ht] -> 
# f[0ht, 2ht, 4ht, 6ht],f[1ht, 3ht, 5ht, 7ht]
# vm0/vm1: _MASK_0022/_MASK_1133
# v0: _MASK_V0_0b0101
.macro shuffle_x2 in0_0, in0_1, in1_0, in1_1, \
        tm0_0, tm0_1, vm0, vm1
    # shuffle2: tm0_0=f[4ht, 5ht, 4ht, 5ht], tm0_1=f[2ht, 3ht, 2ht, 3ht]
    # shuffle1: tm0_0=f[2ht, 2ht, 6ht, 6ht], tm0_1=f[1ht, 1ht, 5ht, 5ht]
    vrgather.vv \tm0_0, \in0_1, \vm0;       vrgather.vv \tm0_1, \in0_0, \vm1
    # shuffle2: in0_0=f[0ht, 1ht, 4ht, 5ht], in0_1=f[2ht, 3ht, 6ht, 7ht]
    # shuffle1: in0_0=f[0ht, 2ht, 4ht, 6ht], in0_1=f[1ht, 3ht, 5ht, 7ht]
    vmerge.vvm  \in0_0, \tm0_0, \in0_0, v0; vmerge.vvm  \in0_1, \in0_1, \tm0_1, v0
    vrgather.vv \tm0_0, \in1_1, \vm0;       vrgather.vv \tm0_1, \in1_0, \vm1
    vmerge.vvm  \in1_0, \tm0_0, \in1_0, v0; vmerge.vvm  \in1_1, \in1_1, \tm0_1, v0
.endm

.macro CT_BF are,bre,aim,bim,wre,wim,vf,vt0,vt1
	vfmul.v\()\vf \vt0, \bre, \wre
	vfmul.v\()\vf \vt1, \bim, \wim
	vfmul.v\()\vf \bre, \bre, \wim
	vfmul.v\()\vf \bim, \bim, \wre
	vfsub.vv \vt0, \vt0, \vt1
	vfadd.vv \vt1, \bre, \bim
	vfsub.vv \bre, \are, \vt0
	vfsub.vv \bim, \aim, \vt1
	vfadd.vv \are, \are, \vt0
	vfadd.vv \aim, \aim, \vt1
.endm

.macro CT_BFx2 a0re,b0re,a1re,b1re,a0im,b0im,a1im,b1im,\
		w0re,w1re,w0im,w1im,vf,vt00,vt01,vt10,vt11
	vfmul.v\()\vf \vt00, \b0re, \w0re
	vfmul.v\()\vf \vt01, \b0im, \w0im
	vfmul.v\()\vf \vt10, \b1re, \w1re
	vfmul.v\()\vf \vt11, \b1im, \w1im
	vfmul.v\()\vf \b0re, \b0re, \w0im
	vfmul.v\()\vf \b0im, \b0im, \w0re
	vfmul.v\()\vf \b1re, \b1re, \w1im
	vfmul.v\()\vf \b1im, \b1im, \w1re
	vfsub.vv \vt00, \vt00, \vt01
	vfsub.vv \vt10, \vt10, \vt11
	vfadd.vv \vt01, \b0re, \b0im
	vfadd.vv \vt11, \b1re, \b1im
	vfsub.vv \b0re, \a0re, \vt00
	vfsub.vv \b1re, \a1re, \vt10
	vfsub.vv \b0im, \a0im, \vt01
	vfsub.vv \b1im, \a1im, \vt11
	vfadd.vv \a0re, \a0re, \vt00
	vfadd.vv \a1re, \a1re, \vt10
	vfadd.vv \a0im, \a0im, \vt01
	vfadd.vv \a1im, \a1im, \vt11
.endm

# void fpoly_FFT_rvv(unsigned logn, double *f, double *GM);
# t0: n, t1: hn, t2: byte_stride=ht*8, t3: loop number, t4: 2*byte_stride
### For layers 1234:
# ht=32 for Falcon1024, =16 for Falcon512
# byte_stride = n >> 2 (ht*8 for Falcon1024, ht*8 for Falcon512)
# loop number = hn/32 = hn >> 5
### For layers 5678:
FUNC fpoly_FFT_rvv
    addi sp, sp, -8*8
    save_regs
    // TODO: 可以稍微调整以改进流水线
    li t6, 1
    sll  t0, t6, a0 # n = 1 << logn
    srli t1, t0, 1  # hn = n >> 1
    srli t2, t0, 2  # byte_stride=ht*8
    srli t3, t1, 5  # loop number
    slli t4, t2, 1  # 2*byte_stride
    # prepare the base addresses required by vlse64.v
    # a1: &f[0], a3: &f[1], a4: &f[0+hn], a5: &f[1+hn]
    slli t6, t1, 3  # 8*hn
    addi a3, a1, 8
    add  a4, a1, t6
    add  a5, a3, t6
    # prepare the base addresses required by vsse64.v
    # s0: &f[0],   s1: &f[1ht],   s2: &f[8ht],   s3: &f[9ht]
    # s4: &f[0+hn],s5: &f[1ht+hn],s6: &f[8ht+hn],s7: &f[9ht+hn]
    slli t5, t2, 3  # 8*byte_stride
    mv   s0, a1
    add  s1, a1, t2
    add  s2, s0, t5
    add  s3, s1, t5
    add  s4, s0, t6
    add  s5, s1, t6
    add  s6, s2, t6
    add  s7, s3, t6
    vsetivli a7, 4, e64, m1, tu, mu
	### GM: 
    # f0|f1 for layer 1.       v2-v3  |v4-v5   for layer 2.
    # v6-v7|v8-v9 for layer 3. v10-v11|v12-v13 for layer 4.
    fld f0, 0*8(a2);    fld f1, 1*8(a2)
    addi a6, a2, 2*8;   addi a7, a2, 2*8+4*8
    addi t5, a6, 2*32;  addi t6, a7, 2*32
    vle64.v v2, (a6);   vle64.v v4, (a7)
    vle64.v v3, (t5);   vle64.v v5, (t6)
    addi a6, a6, 4*32;  addi a7, a7, 4*32
    addi t5, t5, 4*32;  addi t6, t6, 4*32
    vle64.v v6, (a6);   vle64.v v8, (a7)
    vle64.v v7, (t5);   vle64.v v9, (t6)
    addi a6, a6, 4*32;  addi a7, a7, 4*32
    addi t5, t5, 4*32;  addi t6, t6, 4*32
    vle64.v v10, (a6);  vle64.v v12, (a7)
    vle64.v v11, (t5) 
    # We load v13 when needed in the loop, so that v13 can 
    # be used as a temporary register
    addi a2, t6, 1*32 # jump to the beginning of 5th layer
# 1 2 3 4 layers merging
fpoly_FFT_rvv_1234_loop:
    vsetivli a7, 16, e64, m4, tu, mu
	# v16-v19=[f[0],f[ht],  ...,f[15ht]]    | v24-v27 are imaginary part
    # v20-v23=[f[1],f[ht+1],...,f[15ht+1]]  | v28-v31 are imaginary part
	vlse64.v v16, (a1), t2
	vlse64.v v20, (a3), t2
	vlse64.v v24, (a4), t2
	vlse64.v v28, (a5), t2
    vsetivli a7, 4, e64, m1, tu, mu
    # layer 1
    CT_BFx2 v16,v18,v17,v19,v24,v26,v25,v27,f0,f0,f1,f1,f,v0,v1,v14,v15
    CT_BFx2 v20,v22,v21,v23,v28,v30,v29,v31,f0,f0,f1,f1,f,v0,v1,v14,v15
    # layer 2
    CT_BFx2 v16,v17,v18,v19,v24,v25,v26,v27,v2,v3,v4,v5,v,v0,v1,v14,v15
    CT_BFx2 v20,v21,v22,v23,v28,v29,v30,v31,v2,v3,v4,v5,v,v0,v1,v14,v15
    # before shuffle2:
    # v16=f[0ht,1ht, 2ht, 3ht], v17=f[4ht, 5ht, 6ht, 7ht]
    # v18=f[8ht,9ht,10ht,11ht], v19=f[12ht,13ht,14ht,15ht]
    # after shuffle2:
    # v16=f[0ht,1ht, 4ht, 5ht], v17=f[2ht, 3ht, 6ht, 7ht]
    # v18=f[8ht,9ht,12ht,13ht], v19=f[10ht,11ht,14ht,15ht]
    la a6, _MASK_0101;  la a7, _MASK_2323
    li t0, 0b0011
    vle64.v v1, (a6);   vle64.v v13, (a7)
    vmv.s.x v0, t0
    shuffle_x2 v16, v17, v18, v19, v14, v15, v1, v13
    shuffle_x2 v20, v21, v22, v23, v14, v15, v1, v13
    shuffle_x2 v24, v25, v26, v27, v14, v15, v1, v13
    shuffle_x2 v28, v29, v30, v31, v14, v15, v1, v13
    # layer 3
    CT_BFx2 v16,v17,v18,v19,v24,v25,v26,v27,v6,v7,v8,v9,v,v0,v1,v14,v15
    CT_BFx2 v20,v21,v22,v23,v28,v29,v30,v31,v6,v7,v8,v9,v,v0,v1,v14,v15
    # after shuffle1:
    # v16=f[0ht,2ht, 4ht,  6ht], v17=[1ht,3ht, 5ht, 7ht]
    # v18=f[8ht,10ht,12ht,14ht], v19=[9ht,11ht,13ht,15ht]
    la a6, _MASK_0022;  la a7, _MASK_1133
    li t0, 0b0101
    vle64.v v1, (a6);   vle64.v v13, (a7)
    vmv.s.x v0, t0
    shuffle_x2 v16, v17, v18, v19, v14, v15, v1, v13
    shuffle_x2 v20, v21, v22, v23, v14, v15, v1, v13
    shuffle_x2 v24, v25, v26, v27, v14, v15, v1, v13
    shuffle_x2 v28, v29, v30, v31, v14, v15, v1, v13
    # layer 4
    vle64.v v13, (t6)
    CT_BFx2 v16,v17,v18,v19,v24,v25,v26,v27,v10,v11,v12,v13,v,v0,v1,v14,v15
    CT_BFx2 v20,v21,v22,v23,v28,v29,v30,v31,v10,v11,v12,v13,v,v0,v1,v14,v15
    # stride: ht*16 (in bytes)
    # base address: 
    # v16: &f[0ht+0], v17: &f[1ht+0], v18: &f[8ht+0], v19: &f[9ht+0]
    # v24: &f[0ht+hn],v25: &f[1ht+hn],v26: &f[8ht+hn],v27: &f[9ht+hn]
    # v20: &f[0ht+1],   v21: &f[1ht+1],   v22: &f[8ht+1],   v23: &f[9ht+1]
    # v28: &f[0ht+1+hn],v29: &f[1ht+1+hn],v30: &f[8ht+1+hn],v31: &f[9ht+1+hn]
	vsse64.v v16, (s0), t4;  vsse64.v v17, (s1), t4
    vsse64.v v18, (s2), t4;  vsse64.v v19, (s3), t4
    vsse64.v v24, (s4), t4;  vsse64.v v25, (s5), t4
    vsse64.v v26, (s6), t4;  vsse64.v v27, (s7), t4
    addi s0, s0, 8; addi s1, s1, 8
    addi s2, s2, 8; addi s3, s3, 8
    addi s4, s4, 8; addi s5, s5, 8
    addi s6, s6, 8; addi s7, s7, 8
    vsse64.v v20, (s0), t4;  vsse64.v v21, (s1), t4
    vsse64.v v22, (s2), t4;  vsse64.v v23, (s3), t4
    vsse64.v v28, (s4), t4;  vsse64.v v29, (s5), t4
    vsse64.v v30, (s6), t4;  vsse64.v v31, (s7), t4
    addi s0, s0, 8; addi s1, s1, 8
    addi s2, s2, 8; addi s3, s3, 8
    addi s4, s4, 8; addi s5, s5, 8
    addi s6, s6, 8; addi s7, s7, 8
	addi t3, t3, -1
    addi a1, a1, 2*8; addi a3, a3, 2*8
    addi a4, a4, 2*8; addi a5, a5, 2*8
	bnez t3, fpoly_FFT_rvv_1234_loop
    restore_regs
    addi sp, sp, 8*8
FUNC_END fpoly_FFT_rvv

.endif # RVV_VLEN256 == 1
