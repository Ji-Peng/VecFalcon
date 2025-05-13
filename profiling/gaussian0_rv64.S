.section .rodata
.align 3
GAUSS_LO64:
  .dword 17866957108348000258
  .dword 15216282288489618306
  .dword 9065130955956142591
  .dword 15093043907930966756
  .dword 10773855707238178671
  .dword 8595902006365044063
  .dword 1163297957344668388
  .dword 117656387352093658
  .dword 8867391802663976
  .dword 496969357462633
  .dword 20680885154299
  .dword 638331848991
  .dword 14602316184
  .dword 247426747
  .dword 3104126
  .dword 28824

.align 4
GAUSS0_RVV:
.word 3741698,3741698,3741698,3741698,3741698,3741698,3741698,3741698
.word 3068844,3068844,3068844,3068844,3068844,3068844,3068844,3068844
.word 10745844,10745844,10745844,10745844,10745844,10745844,10745844,10745844
.word 8248194,8248194,8248194,8248194,8248194,8248194,8248194,8248194
.word 1580863,1580863,1580863,1580863,1580863,1580863,1580863,1580863
.word 5559083,5559083,5559083,5559083,5559083,5559083,5559083,5559083
.word 2736639,2736639,2736639,2736639,2736639,2736639,2736639,2736639
.word 13669192,13669192,13669192,13669192,13669192,13669192,13669192,13669192
.word 2260429,2260429,2260429,2260429,2260429,2260429,2260429,2260429
.word 10046180,10046180,10046180,10046180,10046180,10046180,10046180,10046180
.word 4421575,4421575,4421575,4421575,4421575,4421575,4421575,4421575
.word 708981,708981,708981,708981,708981,708981,708981,708981
.word 4136815,4136815,4136815,4136815,4136815,4136815,4136815,4136815
.word 7122675,7122675,7122675,7122675,7122675,7122675,7122675,7122675
.word 169348,169348,169348,169348,169348,169348,169348,169348
.word 7650655,7650655,7650655,7650655,7650655,7650655,7650655,7650655
.word 13063405,13063405,13063405,13063405,13063405,13063405,13063405,13063405
.word 30538,30538,30538,30538,30538,30538,30538,30538
# GAUSS0_RVV[6][0]:
.word 7826148
.word 14505003
.word 4132
.word 11363290
.word 16768101
.word 417
.word 8086568
.word 8444042
.word 31
.word 265321
.word 12844466
.word 1
# GAUSS0_RVV[10]:
.word 13644283
.word 1232676
.word 9111839
.word 38047
.word 6138264
.word 870
.word 12545723
.word 14
# GAUSS0_RVV[14]:
.word 3104126
.word 28824
.word 198
.word 1

.text

.macro save_regs
  sd s0,  0*8(sp)
  sd s1,  1*8(sp)
  sd s2,  2*8(sp)
  sd s3,  3*8(sp)
  sd s4,  4*8(sp)
  sd s5,  5*8(sp)
  sd s6,  6*8(sp)
  sd s7,  7*8(sp)
  sd s8,  8*8(sp)
  sd s9,  9*8(sp)
  sd s10, 10*8(sp)
  sd s11, 11*8(sp)
  sd gp,  12*8(sp)
  sd tp,  13*8(sp)
  sd ra,  14*8(sp)
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
  ld s8,  8*8(sp)
  ld s9,  9*8(sp)
  ld s10, 10*8(sp)
  ld s11, 11*8(sp)
  ld gp,  12*8(sp)
  ld tp,  13*8(sp)
  ld ra,  14*8(sp)
.endm

# high 8 bits of the Gaussian0 table
# GAUSS_5_HI8 - GAUSS_17_HI8 are zeros
.equ GAUSS_0_HI8, 163
.equ GAUSS_1_HI8, 84
.equ GAUSS_2_HI8, 34
.equ GAUSS_3_HI8, 10
.equ GAUSS_4_HI8, 2
# lower 64 bits of the gaussian0 table entry that 
# can be encoded as an immediate value
.equ GAUSS_16_LO64, 198
.equ GAUSS_17_LO64, 1

.macro LESS_THAN_72b_3W r0, r1, r2, tmp0, tmp1, tmp2, \
    i0, i1, i2, inh, inl, GA0L, GA1L, GA2L, GA0H, GA1H, GA2H
.if \i0 < 16
    sltu  \tmp0, \inl, \GA0L
.else
    sltiu \tmp0, \inl, \GA0L
.endif
.if \i1 < 16
    sltu  \tmp1, \inl, \GA1L
.else
    sltiu \tmp1, \inl, \GA1L
.endif
.if \i2 < 16
    sltu  \tmp2, \inl, \GA2L
.else
    sltiu \tmp2, \inl, \GA2L
.endif
    sub  \r0, \inh, \tmp0
    sub  \r1, \inh, \tmp1
    sub  \r2, \inh, \tmp2
.if \i0 < 5
    addi \r0, \r0, -\GA0H
.endif
.if \i1 < 5
    addi \r1, \r1, -\GA1H
.endif
.if \i2 < 5
    addi \r2, \r2, -\GA2H
.endif
    srli \r0, \r0, 63
    srli \r1, \r1, 63
    srli \r2, \r2, 63
.endm

.macro LESS_THAN_72b_3W_RVV r0, r1, r2, vt0, vt1, \
    i0, i1, i2, inh, inm, inl, GA0L, GA0M, GA0H, \
    GA1L, GA1M, GA1H, GA2L, GA2M, GA2H

.if \i0 <= 5
    vmsbc.vv  v0,   \inl, \GA0L         // low 24b 0th
.else
    vmsbc.vx  v0,   \inl, \GA0L         // low 24b 0th
.endif
.if \i1 <= 5
    vsub.vv   \r1, \inl, \GA1L          // low 24b 1th step 0
.else
    vsub.vx   \r1, \inl, \GA1L          // low 24b 1th step 0
.endif
.if \i2 <= 5
    vsub.vv   \r2, \inl, \GA2L          // low 24b 2th step 0
.else
    vsub.vx   \r2, \inl, \GA2L          // low 24b 2th step 0
.endif
.if \i1 <= 5
    vsub.vv   \vt0, \inm, \GA1M         // mid 24b 1th step 0
.else
    .if \i1 <= 13
    vsub.vx   \vt0, \inm, \GA1M         // mid 24b 1th step 0
    .endif
.endif
.if \i2 <= 5
    vsub.vv   \vt1, \inm, \GA2M         // mid 24b 2th step 0
.else
    .if \i2 <= 13
    vsub.vx   \vt1, \inm, \GA2M         // mid 24b 2th step 0
    .endif
.endif
    vsrl.vi   \r1, \r1, 31              // low 24b 1th step 1
.if \i0 <= 5
    vmsbc.vvm v0, \inm, \GA0M, v0	    // mid 24b 0th
.else
    .if \i0 <= 13
    vmsbc.vxm v0, \inm, \GA0M, v0	    // mid 24b 0th
    .else
    vmsbc.vxm v0, \inm, zero, v0	    // mid 24b 0th
    .endif
.endif
    vsrl.vi   \r2, \r2, 31              // low 24b 2th step 1
.if \i1 <= 13
    vsub.vv   \vt0, \vt0, \r1           // mid 24b 1th step 1
.else
    vsub.vv   \vt0, \inm, \r1           // mid 24b 1th step 1
.endif
.if \i2 <= 13
    vsub.vv   \vt1, \vt1, \r2           // mid 24b 2th step 1
.else
    vsub.vv   \vt1, \inm, \r2           // mid 24b 2th step 1
.endif
.if \i1 <= 5
    vsub.vv   \r1, \inh, \GA1H          // high 24b 1th step 0
.else
    .if \i1 <= 9
    vsub.vx   \r1, \inh, \GA1H          // high 24b 1th step 0
    .endif
.endif
.if \i2 <= 5
    vsub.vv   \r2, \inh, \GA2H          // high 24b 2th step 0
.else
    .if \i2 <= 9
    vsub.vx   \r2, \inh, \GA2H          // high 24b 2th step 0
    .endif
.endif
    vsrl.vi   \vt0, \vt0, 31            // mid 24b 1th step 2
.if \i0 <= 5
    vmsbc.vvm v0, \inh, \GA0H, v0	    // high 24b 0th
.else
    .if \i0 <= 9
    vmsbc.vxm v0, \inh, \GA0H, v0	    // high 24b 0th
    .else
    vmsbc.vxm v0, \inh, zero, v0	    // high 24b 0th
    .endif
.endif
    vsrl.vi   \vt1, \vt1, 31            // mid 24b 2th step 2
.if \i1 <= 9
    vsub.vv   \r1, \r1, \vt0            // high 24b 1th step 1
.else
    vsub.vv   \r1, \inh, \vt0           // high 24b 1th step 1
.endif
.if \i2 <= 9
    vsub.vv   \r2, \r2, \vt1            // high 24b 2th step 1
.else
    vsub.vv   \r2, \inh, \vt1           // high 24b 2th step 1
.endif
    vsrl.vi   \r1, \r1, 31              // high 24b 1th step 2
    vadd.vi   \r0, \r0, 1, v0.t         // vr0+=1 with mask
    vsrl.vi   \r2, \r2, 31              // high 24b 2th step 2
.endm

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

# register usage:
# t0-t6, a3-a7, s8-s11: RCDT used by gaussian0 sampler
# remaining registers: a0-a2, s0-s7, gp, tp, ra
# a0: pointer to output
# a1: pointer to input
FUNC gaussian0_rv64im
    addi sp, sp, -15*8
    save_regs
    # load RCDT
    la ra, GAUSS_LO64
    ld t0,  0*8(ra)
    ld t1,  1*8(ra)
    ld t2,  2*8(ra)
    ld t3,  3*8(ra)
    ld t4,  4*8(ra)
    ld t5,  5*8(ra)
    ld t6,  6*8(ra)
    ld a3,  7*8(ra)
    ld a4,  8*8(ra)
    ld a5,  9*8(ra)
    ld a6,  10*8(ra)
    ld a7,  11*8(ra)
    ld s8,  12*8(ra)
    ld s9,  13*8(ra)
    ld s10, 14*8(ra)
    ld s11, 15*8(ra)
    # load input 72-bit pseudo-random number
    ld s0, 0(a1)
    ld s1, 8(a1)

    LESS_THAN_72b_3W gp, tp, s7, s6, s5, s4, 0, 1, 2, \
        s1, s0, t0, t1, t2, GAUSS_0_HI8, GAUSS_1_HI8, GAUSS_2_HI8
    add s2, gp, s7
    LESS_THAN_72b_3W gp, ra, s7, s6, s5, s4, 3, 4, 5, \
        s1, s0, t3, t4, t5, GAUSS_3_HI8, GAUSS_4_HI8, GAUSS_5_HI8
    add s2, s2, tp
    add ra, ra, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, tp, s7, s6, s5, s4, 6, 7, 8, \
        s1, s0, t6, a3, a4, GAUSS_6_HI8, GAUSS_7_HI8, GAUSS_8_HI8
    add s2, s2, ra
    add tp, tp, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, ra, s7, s6, s5, s4, 9, 10, 11, \
        s1, s0, a5, a6, a7, GAUSS_9_HI8, GAUSS_10_HI8, GAUSS_11_HI8
    add s2, s2, tp
    add ra, ra, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, tp, s7, s6, s5, s4, 12, 13, 14, \
        s1, s0, s8, s9, s10, GAUSS_12_HI8, GAUSS_13_HI8, GAUSS_14_HI8
    add s2, s2, ra
    add tp, tp, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, ra, s7, s6, s5, s4, 15, 16, 17, \
        s1, s0, s11, GAUSS_16_LO64, GAUSS_17_LO64, \
        GAUSS_15_HI8, GAUSS_16_HI8, GAUSS_17_HI8
    add s2, s2, tp
    add ra, ra, s7
    add s2, s2, gp
    add s2, s2, ra

    sw s2, 0(a0)

    restore_regs
    addi sp, sp, 15*8
FUNC_END gaussian0_rv64im

# register usage:
# t0-t6, a3-a7, s8-s11: RCDT used by gaussian0 sampler
# remaining registers: a0-a2, s0-s7, gp, tp, ra
# a0: pointer to output
# a1: pointer to input
# a2: n-way loop counter
FUNC gaussian0_rv64im_nw
    addi sp, sp, -15*8
    save_regs
    # load RCDT
    la ra, GAUSS_LO64
    ld t0,  0*8(ra)
    ld t1,  1*8(ra)
    ld t2,  2*8(ra)
    ld t3,  3*8(ra)
    ld t4,  4*8(ra)
    ld t5,  5*8(ra)
    ld t6,  6*8(ra)
    ld a3,  7*8(ra)
    ld a4,  8*8(ra)
    ld a5,  9*8(ra)
    ld a6,  10*8(ra)
    ld a7,  11*8(ra)
    ld s8,  12*8(ra)
    ld s9,  13*8(ra)
    ld s10, 14*8(ra)
    ld s11, 15*8(ra)
gaussian0_rv64im_nw_loop:
    # load input 72-bit pseudo-random number
    ld s0, 0(a1)
    ld s1, 8(a1)
    addi a1, a1, 2*8
    LESS_THAN_72b_3W gp, tp, s7, s6, s5, s4, 0, 1, 2, \
        s1, s0, t0, t1, t2, GAUSS_0_HI8, GAUSS_1_HI8, GAUSS_2_HI8
    add s2, gp, s7
    LESS_THAN_72b_3W gp, ra, s7, s6, s5, s4, 3, 4, 5, \
        s1, s0, t3, t4, t5, GAUSS_3_HI8, GAUSS_4_HI8, GAUSS_5_HI8
    add s2, s2, tp
    add ra, ra, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, tp, s7, s6, s5, s4, 6, 7, 8, \
        s1, s0, t6, a3, a4, GAUSS_6_HI8, GAUSS_7_HI8, GAUSS_8_HI8
    add s2, s2, ra
    add tp, tp, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, ra, s7, s6, s5, s4, 9, 10, 11, \
        s1, s0, a5, a6, a7, GAUSS_9_HI8, GAUSS_10_HI8, GAUSS_11_HI8
    add s2, s2, tp
    add ra, ra, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, tp, s7, s6, s5, s4, 12, 13, 14, \
        s1, s0, s8, s9, s10, GAUSS_12_HI8, GAUSS_13_HI8, GAUSS_14_HI8
    add s2, s2, ra
    add tp, tp, s7
    add s2, s2, gp
    LESS_THAN_72b_3W gp, ra, s7, s6, s5, s4, 15, 16, 17, \
        s1, s0, s11, GAUSS_16_LO64, GAUSS_17_LO64, \
        GAUSS_15_HI8, GAUSS_16_HI8, GAUSS_17_HI8
    add s2, s2, tp
    add ra, ra, s7
    add s2, s2, gp
    addi a2, a2, -1
    add s2, s2, ra
    # store output
    sw  s2, 0(a0)
    addi a0, a0, 4
    bne a2, zero, gaussian0_rv64im_nw_loop

    restore_regs
    addi sp, sp, 15*8
FUNC_END gaussian0_rv64im_nw

.if RVV_VLEN256 == 1
# register usage:
# a0: pointer to output
# a1: pointer to input
# a2: n-way loop counter
# a3, t0, t1: tmp registers
# v16-v31: RCDT
# t2-t6, s0-s11, a4-a7, gp, ra, tp: RCDT
FUNC gaussian0_rvv
    addi sp, sp, -15*8
    save_regs

    vsetivli t0, 8, e32, m1, tu, mu
    # load RCDT
    la ra, GAUSS0_RVV
    la tp, GAUSS0_RVV
    slli t0, t0, 2
    addi ra, ra, 4*8
    vle32.v v16, (tp);   addi tp, tp, 4*8*2 # [0][0]
    vle32.v v17, (ra);   addi ra, ra, 4*8*2 # [0][1]
    vle32.v v18, (tp);   addi tp, tp, 4*8*2 # [0][2]
    vle32.v v19, (ra);   addi ra, ra, 4*8*2 # [1][0]
    vle32.v v20, (tp);   addi tp, tp, 4*8*2 # [1][1] 
    vle32.v v21, (ra);   addi ra, ra, 4*8*2 # [1][2]
    vle32.v v22, (tp);   addi tp, tp, 4*8*2 # [2][0]
    vle32.v v23, (ra);   addi ra, ra, 4*8*2 # [2][1]
    vle32.v v24, (tp);   addi tp, tp, 4*8*2 # [2][2]
    vle32.v v25, (ra);   addi ra, ra, 4*8*2 # [3][0]
    vle32.v v26, (tp);   addi tp, tp, 4*8*2 # [3][1]
    vle32.v v27, (ra);   addi ra, ra, 4*8*2 # [3][2]
    vle32.v v28, (tp);   addi tp, tp, 4*8*2 # [4][0]
    vle32.v v29, (ra);   addi ra, ra, 4*8*2 # [4][1]
    vle32.v v30, (tp);   addi tp, tp, 4*8*2 # [4][2]
    vle32.v v31, (ra);   addi ra, ra, 4*8*2 # [5][0]
    vle32.v v15, (tp);   addi tp, tp, 4*8*2 # [5][1]
    vle32.v v14, (ra)                       # [5][2]
    lw t2,  0*4(tp)                         # [6][0]
    lw t3,  1*4(tp)                         # [6][1]
    lw t4,  2*4(tp)                         # [6][2]
    lw t5,  3*4(tp)                         # [7][0]
    lw t6,  4*4(tp)                         # [7][1]
    lw s0,  5*4(tp)                         # [7][2]
    lw s1,  6*4(tp)                         # [8][0]
    lw s2,  7*4(tp)                         # [8][1]
    lw s3,  8*4(tp)                         # [8][2]
    lw s4,  9*4(tp)                         # [9][0]
    lw s5, 10*4(tp)                         # [9][1]
    lw s6, 11*4(tp)                         # [9][2]
    lw s7, 12*4(tp)                         # [10][0]
    lw s8, 13*4(tp)                         # [10][1]
    lw s9, 14*4(tp)                         # [11][0]
    lw s10,15*4(tp)                         # [11][1]
    lw s11,16*4(tp)                         # [12][0]
    lw a4, 17*4(tp)                         # [12][1]
    lw a5, 18*4(tp)                         # [13][0]
    lw a6, 19*4(tp)                         # [13][1]
    lw a7, 20*4(tp)                         # [14][0]
    lw gp, 21*4(tp)                         # [15][0]
    lw ra, 22*4(tp)                         # [16][0]
    lw tp, 23*4(tp)                         # [17][0]
gaussian0_rvv_loop:
    # load input 72-bit pseudo-random number
    add t1, a1, t0
    vle32.v v2, (a1);   add a1, t1, t0
    vle32.v v3, (t1)
    vxor.vv v5, v5, v5
    vle32.v v4, (a1)
    add a1, a1, t0

    LESS_THAN_72b_3W_RVV v5, v6, v7, v12, v13, 0, 1, 2, \
        v4, v3, v2, v16, v17, v18, v19, v20, v21, \
        v22, v23, v24
    LESS_THAN_72b_3W_RVV v5, v8, v9, v12, v13, 3, 4, 5, \
        v4, v3, v2, v25, v26, v27, v28, v29, v30, \
        v31, v15, v14
    vadd.vv v10, v6, v7
    LESS_THAN_72b_3W_RVV v5, v6, v7, v12, v13, 6, 7, 8, \
        v4, v3, v2, t2, t3, t4, t5, t6, s0, \
        s1, s2, s3
    vadd.vv v11, v8, v9
    LESS_THAN_72b_3W_RVV v5, v8, v9, v12, v13, 9, 10, 11, \
        v4, v3, v2, s4, s5, s6, s7, s8, xx, \
        s9, s10, xx
    vadd.vv v10, v10, v6
    vadd.vv v11, v11, v7
    LESS_THAN_72b_3W_RVV v5, v6, v7, v12, v13, 12, 13, 14, \
        v4, v3, v2, s11, a4, xx, a5, a6, xx, \
        a7, xx, xx
    vadd.vv v10, v10, v8
    vadd.vv v11, v11, v9
    LESS_THAN_72b_3W_RVV v5, v8, v9, v12, v13, 15, 16, 17, \
        v4, v3, v2, gp, xx, xx, ra, xx, xx, \
        tp, xx, xx
    vadd.vv v10, v10, v6
    vadd.vv v11, v11, v7
    vadd.vv v8,  v8,  v9
    vadd.vv v5,  v5,  v10
    vadd.vv v8,  v8,  v11
    addi a2, a2, -1
    vadd.vv v5,  v5,  v8
    # store output
    vse32.v v5, (a0)
    add  a0, a0, t0
    bne  a2, zero, gaussian0_rvv_loop

    restore_regs
    addi sp, sp, 15*8
FUNC_END gaussian0_rvv

# register usage:
# a0: pointer to output (z_bimodal)
# a1: pointer to output (z_square)
# a2: pointer to input (prn for generating z)
# a3: pointer to input (prn for tunning bi and sq)
# a4: n-way loop counter
# t0: tmp registers
# v16-v31: RCDT
# t2-t6, s0-s11, a5-a7, gp, ra, tp: RCDT
FUNC gaussian0_rvv_bisq
    addi sp, sp, -15*8
    save_regs

    vsetivli t0, 8, e32, m1, tu, mu
    # load RCDT
    la ra, GAUSS0_RVV
    la tp, GAUSS0_RVV
    slli t0, t0, 2
    addi ra, ra, 4*8
    vle32.v v16, (tp);   addi tp, tp, 4*8*2 # [0][0]
    vle32.v v17, (ra);   addi ra, ra, 4*8*2 # [0][1]
    vle32.v v18, (tp);   addi tp, tp, 4*8*2 # [0][2]
    vle32.v v19, (ra);   addi ra, ra, 4*8*2 # [1][0]
    vle32.v v20, (tp);   addi tp, tp, 4*8*2 # [1][1] 
    vle32.v v21, (ra);   addi ra, ra, 4*8*2 # [1][2]
    vle32.v v22, (tp);   addi tp, tp, 4*8*2 # [2][0]
    vle32.v v23, (ra);   addi ra, ra, 4*8*2 # [2][1]
    vle32.v v24, (tp);   addi tp, tp, 4*8*2 # [2][2]
    vle32.v v25, (ra);   addi ra, ra, 4*8*2 # [3][0]
    vle32.v v26, (tp);   addi tp, tp, 4*8*2 # [3][1]
    vle32.v v27, (ra);   addi ra, ra, 4*8*2 # [3][2]
    vle32.v v28, (tp);   addi tp, tp, 4*8*2 # [4][0]
    vle32.v v29, (ra);   addi ra, ra, 4*8*2 # [4][1]
    vle32.v v30, (tp);   addi tp, tp, 4*8*2 # [4][2]
    vle32.v v31, (ra);   addi ra, ra, 4*8*2 # [5][0]
    vle32.v v15, (tp);   addi tp, tp, 4*8*2 # [5][1]
    vle32.v v14, (ra)                       # [5][2]
    lw t2,  0*4(tp)                         # [6][0]
    lw t3,  1*4(tp)                         # [6][1]
    lw t4,  2*4(tp)                         # [6][2]
    lw t5,  3*4(tp)                         # [7][0]
    lw t6,  4*4(tp)                         # [7][1]
    lw s0,  5*4(tp)                         # [7][2]
    lw s1,  6*4(tp)                         # [8][0]
    lw s2,  7*4(tp)                         # [8][1]
    lw s3,  8*4(tp)                         # [8][2]
    lw s4,  9*4(tp)                         # [9][0]
    lw s5, 10*4(tp)                         # [9][1]
    lw s6, 11*4(tp)                         # [9][2]
    lw s7, 12*4(tp)                         # [10][0]
    lw s8, 13*4(tp)                         # [10][1]
    lw s9, 14*4(tp)                         # [11][0]
    lw s10,15*4(tp)                         # [11][1]
    lw s11,16*4(tp)                         # [12][0]
    lw t1, 17*4(tp)                         # [12][1]
    lw a5, 18*4(tp)                         # [13][0]
    lw a6, 19*4(tp)                         # [13][1]
    lw a7, 20*4(tp)                         # [14][0]
    lw gp, 21*4(tp)                         # [15][0]
    lw ra, 22*4(tp)                         # [16][0]
    lw tp, 23*4(tp)                         # [17][0]
gaussian0_rvv_bisq_loop:
    # load input 72-bit pseudo-random number
    vle32.v v2, (a2);   add a2, a2, t0
    vle32.v v3, (a2);   add a2, a2, t0
    vxor.vv v5, v5, v5
    vle32.v v4, (a2);   add a2, a2, t0
    LESS_THAN_72b_3W_RVV v5, v6, v7, v12, v13, 0, 1, 2, \
        v4, v3, v2, v16, v17, v18, v19, v20, v21, \
        v22, v23, v24
    LESS_THAN_72b_3W_RVV v5, v8, v9, v12, v13, 3, 4, 5, \
        v4, v3, v2, v25, v26, v27, v28, v29, v30, \
        v31, v15, v14
    vadd.vv v10, v6, v7
    LESS_THAN_72b_3W_RVV v5, v6, v7, v12, v13, 6, 7, 8, \
        v4, v3, v2, t2, t3, t4, t5, t6, s0, \
        s1, s2, s3
    vadd.vv v11, v8, v9
    LESS_THAN_72b_3W_RVV v5, v8, v9, v12, v13, 9, 10, 11, \
        v4, v3, v2, s4, s5, s6, s7, s8, xx, \
        s9, s10, xx
    vadd.vv v10, v10, v6
    vadd.vv v11, v11, v7
    LESS_THAN_72b_3W_RVV v5, v6, v7, v12, v13, 12, 13, 14, \
        v4, v3, v2, s11, t1, xx, a5, a6, xx, \
        a7, xx, xx
    vadd.vv v10, v10, v8
    vadd.vv v11, v11, v9
    LESS_THAN_72b_3W_RVV v5, v8, v9, v12, v13, 15, 16, 17, \
        v4, v3, v2, gp, xx, xx, ra, xx, xx, \
        tp, xx, xx
    vadd.vv v10, v10, v6
    vadd.vv v11, v11, v7
    vle32.v v2, (a2);   add a2, a2, t0
    vle32.v v3, (a2);   add a2, a2, t0
    vadd.vv v8,  v8,  v9
    vadd.vv v5,  v5,  v10
    vxor.vv v6, v6, v6
    vle32.v v4, (a2);   add a2, a2, t0
    vadd.vv v8,  v8,  v11
    vadd.vv v5,  v5,  v8
    LESS_THAN_72b_3W_RVV v6, v7, v8, v12, v13, 0, 1, 2, \
        v4, v3, v2, v16, v17, v18, v19, v20, v21, \
        v22, v23, v24
    LESS_THAN_72b_3W_RVV v6, v9, v10,v12, v13, 3, 4, 5, \
        v4, v3, v2, v25, v26, v27, v28, v29, v30, \
        v31, v15, v14
    vadd.vv v11, v7, v8
    LESS_THAN_72b_3W_RVV v6, v7, v8, v12, v13, 6, 7, 8, \
        v4, v3, v2, t2, t3, t4, t5, t6, s0, \
        s1, s2, s3
    vadd.vv v11, v11, v9
    vadd.vv v6,  v6,  v10
    LESS_THAN_72b_3W_RVV v6, v9, v10,v12, v13, 9, 10, 11, \
        v4, v3, v2, s4, s5, s6, s7, s8, xx, \
        s9, s10, xx
    vadd.vv v11, v11, v7
    vadd.vv v6,  v6,  v8
    LESS_THAN_72b_3W_RVV v6, v7, v8, v12, v13, 12, 13, 14, \
        v4, v3, v2, s11, t1, xx, a5, a6, xx, \
        a7, xx, xx
    vadd.vv v11, v11, v9
    vadd.vv v6,  v6,  v10
    LESS_THAN_72b_3W_RVV v6, v9, v10,v12, v13, 15, 16, 17, \
        v4, v3, v2, gp, xx, xx, ra, xx, xx, \
        tp, xx, xx
    vadd.vv v11, v11, v7
    vle32.v v7, (a3);   add a3, a3, t0
    vadd.vv v8,  v8,  v9
    vadd.vv v6,  v6,  v10
    vle32.v v12, (a3);   add a3, a3, t0
    vadd.vv v11, v11, v8
    vsll.vi v8, v7, 1
    vsll.vi v13, v12, 1
    vadd.vv v6,  v6,  v11
    vadd.vi v8, v8, -1
    vadd.vi v13, v13, -1
    vmul.vv v8, v8, v5
    vmul.vv v13, v13, v6
    vmul.vv v9, v5,  v5
    vmul.vv v10, v6,  v6
    vadd.vv v7, v7, v8
    vadd.vv v12, v12, v13
    vse32.v v9, (a1);   add  a1, a1, t0
    vse32.v v7, (a0);   add  a0, a0, t0
    vse32.v v10, (a1);  add  a1, a1, t0
    vse32.v v12, (a0);  add  a0, a0, t0

    addi a4, a4, -2
    bne  a4, zero, gaussian0_rvv_bisq_loop

    restore_regs
    addi sp, sp, 15*8
FUNC_END gaussian0_rvv_bisq

.endif
