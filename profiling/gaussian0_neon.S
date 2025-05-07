.section .rodata

.align 4
GAUSS0_NEON:
// GAUSS0_NEON[0-5] normal order:
.word 3741698,3741698,3741698,3741698           // GAUSS0_NEON[0]
.word 3068844,3068844,3068844,3068844
.word 10745844,10745844,10745844,10745844
.word 8248194,8248194,8248194,8248194           // GAUSS0_NEON[1]
.word 1580863,1580863,1580863,1580863
.word 5559083,5559083,5559083,5559083
.word 2736639,2736639,2736639,2736639           // GAUSS0_NEON[2]
.word 13669192,13669192,13669192,13669192
.word 2260429,2260429,2260429,2260429
.word 10046180,10046180,10046180,10046180       // GAUSS0_NEON[3]
.word 4421575,4421575,4421575,4421575
.word 708981,708981,708981,708981
.word 4136815,4136815,4136815,4136815           // GAUSS0_NEON[4]
.word 7122675,7122675,7122675,7122675
.word 169348,169348,169348,169348
.word 7650655,7650655,7650655,7650655           // GAUSS0_NEON[5]
.word 13063405,13063405,13063405,13063405
.word 30538,30538,30538,30538
.word 7826148,7826148,7826148,7826148           // GAUSS0_NEON[6][0]
.word 11363290,11363290,11363290,11363290       // GAUSS0_NEON[7][0]
.word 8086568,8086568,8086568,8086568           // GAUSS0_NEON[8][0]
.word 14505003,14505003,14505003,14505003       // GAUSS0_NEON[6][1]
.word 16768101,16768101,16768101,16768101       // GAUSS0_NEON[7][1]
.word 8444042,8444042,8444042,8444042           // GAUSS0_NEON[8][1]
.word 4132,4132,4132,4132                       // GAUSS0_NEON[6][2]
.word 417,417,417,417                           // GAUSS0_NEON[7][2]
.word 31,31,31,31                               // GAUSS0_NEON[8][2]
.word 265321,265321,265321,265321               // GAUSS0_NEON[9][0]
.word 13644283,13644283,13644283,13644283       // GAUSS0_NEON[10][0]
.word 9111839,9111839,9111839,9111839           // GAUSS0_NEON[11][0]
.word 12844466,12844466,12844466,12844466       // GAUSS0_NEON[9][1]
.word 1232676,1232676,1232676,1232676           // GAUSS0_NEON[10][1]
.word 38047,38047,38047,38047                   // GAUSS0_NEON[11][1]
.word 1,1,1,1                                   // GAUSS0_NEON[9][2]
.word 6138264,6138264,6138264,6138264           // GAUSS0_NEON[12][0]
.word 12545723,12545723,12545723,12545723       // GAUSS0_NEON[13][0]
.word 3104126,3104126,3104126,3104126           // GAUSS0_NEON[14][0]
.word 870,870,870,870                           // GAUSS0_NEON[12][1]
.word 14,14,14,14                               // GAUSS0_NEON[13][1]
.word 28824,28824,28824,28824                   // GAUSS0_NEON[15][0]
.word 198,198,198,198                           // GAUSS0_NEON[16][0]
.word 1,1,1,1                                   // GAUSS0_NEON[17][0]

.text

.macro LESS_THAN_72b_3W vt0, vt1, vt2, vt3, vt4, vt5, \
    i0, i1, i2, inl, inm, inh, GA0L, GA0M, GA0H, \
    GA1L, GA1M, GA1H, GA2L, GA2M, GA2H, load=0
    cmgt \vt3\().4s, \GA0L\().4s, \inl\().4s     // if inl < GA0L then -1 else 0
    cmgt \vt4\().4s, \GA1L\().4s, \inl\().4s
    cmgt \vt5\().4s, \GA2L\().4s, \inl\().4s
    sub  \vt0\().4s, \inm\().4s, \GA0M\().4s
    sub  \vt1\().4s, \inm\().4s, \GA1M\().4s
    sub  \vt2\().4s, \inm\().4s, \GA2M\().4s
    add  \vt0\().4s, \vt0\().4s, \vt3\().4s
    add  \vt1\().4s, \vt1\().4s, \vt4\().4s
    add  \vt2\().4s, \vt2\().4s, \vt5\().4s
    ushr \vt0\().4s, \vt0\().4s, #31
    ushr \vt1\().4s, \vt1\().4s, #31
    ushr \vt2\().4s, \vt2\().4s, #31
    sub  \vt3\().4s, \inh\().4s, \GA0H\().4s
    sub  \vt4\().4s, \inh\().4s, \GA1H\().4s
    sub  \vt5\().4s, \inh\().4s, \GA2H\().4s
    sub  \vt0\().4s, \vt3\().4s, \vt0\().4s
    sub  \vt1\().4s, \vt4\().4s, \vt1\().4s
    sub  \vt2\().4s, \vt5\().4s, \vt2\().4s
.if \load == 1
    ld1 {v28.4s}, [x6], #16                     // GAUSS0_NEON[8][0]
.endif
    usra v20.4s, \vt0\().4s, #31
    usra v21.4s, \vt1\().4s, #31
    usra v22.4s, \vt2\().4s, #31
.endm

.macro LESS_THAN_72b_3W_V2 vt0, vt1, vt2, vt3, vt4, vt5, \
    i0, i1, i2, inl, inm, inh, GA0L, GA1L, GA2L, ADDR, load=0
.if \i0 == 12
    ld1  {\vt0\().4s, \vt1\().4s}, [\ADDR], #32    // GA0M, GA1M
.else
    .if \i0 == 15
    .else
        ld1  {\vt0\().4s, \vt1\().4s, \vt2\().4s}, [\ADDR], #48    // GA0M, GA1M, GA2M
    .endif
.endif
    cmgt \vt3\().4s, \GA0L\().4s, \inl\().4s     // if inl < GA0L then -1 else 0
    cmgt \vt4\().4s, \GA1L\().4s, \inl\().4s
    cmgt \vt5\().4s, \GA2L\().4s, \inl\().4s
    add  \vt3\().4s, \vt3\().4s, \inm\().4s
    add  \vt4\().4s, \vt4\().4s, \inm\().4s
    add  \vt5\().4s, \vt5\().4s, \inm\().4s
.if \i0 == 12
    sub  \vt3\().4s, \vt3\().4s, \vt0\().4s
    sub  \vt4\().4s, \vt4\().4s, \vt1\().4s
.else
    .if \i0 == 15
    .else
        sub  \vt3\().4s, \vt3\().4s, \vt0\().4s
        sub  \vt4\().4s, \vt4\().4s, \vt1\().4s
        sub  \vt5\().4s, \vt5\().4s, \vt2\().4s
    .endif
.endif
.if \i0 == 9
    ld1  {\vt0\().4s}, [\ADDR], #16    // GA0H
    ushr \vt4\().4s, \vt4\().4s, #31
    ushr \vt5\().4s, \vt5\().4s, #31
    usra \vt0\().4s, \vt3\().4s, #31
    sub  \vt1\().4s, \inh\().4s, \vt4\().4s
    sub  \vt2\().4s, \inh\().4s, \vt5\().4s
    .if \load == 1
        ld1 {v26.4s, v27.4s, v28.4s}, [x6], #48
    .endif
    sub  \vt0\().4s, \inh\().4s, \vt0\().4s
    usra v21.4s, \vt1\().4s, #31
    usra v22.4s, \vt2\().4s, #31
    usra v20.4s, \vt0\().4s, #31
.else
    .if \i0 >= 12
        ushr \vt3\().4s, \vt3\().4s, #31
        ushr \vt4\().4s, \vt4\().4s, #31
        ushr \vt5\().4s, \vt5\().4s, #31
        sub  \vt0\().4s, \inh\().4s, \vt3\().4s
        sub  \vt1\().4s, \inh\().4s, \vt4\().4s
        sub  \vt2\().4s, \inh\().4s, \vt5\().4s
        .if \load == 1
            ld1 {v26.4s, v27.4s, v28.4s}, [x6], #48
        .endif
        usra v20.4s, \vt0\().4s, #31
        usra v21.4s, \vt1\().4s, #31
        usra v22.4s, \vt2\().4s, #31
    .else
        ld1  {\vt0\().4s, \vt1\().4s, \vt2\().4s}, [\ADDR], #48    // GA0H, GA1H, GA2H
        usra \vt0\().4s, \vt3\().4s, #31
        usra \vt1\().4s, \vt4\().4s, #31
        usra \vt2\().4s, \vt5\().4s, #31
        .if \load == 1
            ld1 {v26.4s, v27.4s, v28.4s}, [x6], #48
        .endif
        sub  \vt0\().4s, \inh\().4s, \vt0\().4s
        sub  \vt1\().4s, \inh\().4s, \vt1\().4s
        sub  \vt2\().4s, \inh\().4s, \vt2\().4s
        usra v20.4s, \vt0\().4s, #31
        usra v21.4s, \vt1\().4s, #31
        usra v22.4s, \vt2\().4s, #31
    .endif
.endif
.endm

.macro FUNC name
    .global \name
    .align 4
    \name:
.endm

.macro FUNC_END name
    ret
    .type \name, %function
    .size \name, .-\name
.endm

/*************************************************
* register usage:
* v0  - v19: cache GAUSS0_NEON table
* v20 - v25: Intermediate results to be accumulated
* v26 - v28: temporary registers
* v29 - v31: input prn
void gaussian0_neon(int32_t *z, uint32_t *prn, size_t n_way);
**************************************************/
FUNC gaussian0_neon
    adr x5, GAUSS0_NEON
    // v0 - v17: GAUSS0_NEON[0-5]; v18,v19: GAUSS0_NEON[6,7][0]
    ld1 {v0.4s-v3.4s},   [x5], #64
    ld1 {v4.4s-v7.4s},   [x5], #64
    ld1 {v8.4s-v11.4s},  [x5], #64
    ld1 {v12.4s-v15.4s}, [x5], #64
    ld1 {v16.4s-v19.4s}, [x5], #64
gaussian0_neon_loop:
    ld1 {v29.4s, v30.4s, v31.4s}, [x1], #48     // input prn
    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    mov x6, x5
    LESS_THAN_72b_3W v23, v24, v25, v26, v27, v28, 0, 1, 2, \
        v29, v30, v31, v0, v1, v2, v3, v4, v5, v6, v7, v8, 0
    LESS_THAN_72b_3W v23, v24, v25, v26, v27, v28, 3, 4, 5, \
        v29, v30, v31, v9, v10, v11, v12, v13, v14, v15, v16, v17, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 6, 7, 8, \
        v29, v30, v31, v18, v19, v28, x6, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 9, 10, 11, \
        v29, v30, v31, v26, v27, v28, x6, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 12, 13, 14, \
        v29, v30, v31, v26, v27, v28, x6, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 15, 16, 17, \
        v29, v30, v31, v26, v27, v28, x6, 0
    add v21.4s, v21.4s, v22.4s
    add v20.4s, v20.4s, v21.4s
    sub x2, x2, #1
    st1 {v20.4s}, [x0], #16
    cbnz x2, gaussian0_neon_loop
gaussian0_neon_loop_end:
    ret
FUNC_END gaussian0_neon

/*************************************************
* register usage:
* v0  - v19: cache GAUSS0_NEON table
* v20 - v25: Intermediate results to be accumulated
* v26 - v28: temporary registers
* v29 - v31: input prn
void gaussian0_neon_bisq(int32_t *z_bi, int32_t *z_sq, 
    uint32_t *prn, uint32_t *prn_bisq, size_t n_way);
**************************************************/
FUNC gaussian0_neon_bisq
    adr x5, GAUSS0_NEON
    // v0 - v17: GAUSS0_NEON[0-5]; v18,v19: GAUSS0_NEON[6,7][0]
    ld1 {v0.4s-v3.4s},   [x5], #64
    ld1 {v4.4s-v7.4s},   [x5], #64
    ld1 {v8.4s-v11.4s},  [x5], #64
    ld1 {v12.4s-v15.4s}, [x5], #64
    ld1 {v16.4s-v19.4s}, [x5], #64
gaussian0_neon_bisq_loop:
    ld1 {v29.4s, v30.4s, v31.4s}, [x2], #48     // input prn
    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    mov x6, x5
    LESS_THAN_72b_3W v23, v24, v25, v26, v27, v28, 0, 1, 2, \
        v29, v30, v31, v0, v1, v2, v3, v4, v5, v6, v7, v8, 0
    LESS_THAN_72b_3W v23, v24, v25, v26, v27, v28, 3, 4, 5, \
        v29, v30, v31, v9, v10, v11, v12, v13, v14, v15, v16, v17, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 6, 7, 8, \
        v29, v30, v31, v18, v19, v28, x6, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 9, 10, 11, \
        v29, v30, v31, v26, v27, v28, x6, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 12, 13, 14, \
        v29, v30, v31, v26, v27, v28, x6, 1
    LESS_THAN_72b_3W_V2 v23, v24, v25, v26, v27, v28, 15, 16, 17, \
        v29, v30, v31, v26, v27, v28, x6, 0
    ld1 {v26.4s}, [x3], #16                     // prn for bimodal and square
    add v21.4s, v21.4s, v22.4s
    add v20.4s, v20.4s, v21.4s
    movi v29.4s, #1
    ushl v27.4s, v26.4s, v29.4s
    sub  v27.4s, v27.4s, v29.4s
    sub x4, x4, #1
    mul  v27.4s, v27.4s, v20.4s
    mul  v28.4s, v20.4s, v20.4s
    add  v27.4s, v27.4s, v26.4s
    st1 {v28.4s}, [x1], #16
    st1 {v27.4s}, [x0], #16
    cbnz x4, gaussian0_neon_bisq_loop
gaussian0_neon_bisq_loop_end:
    ret
FUNC_END gaussian0_neon_bisq
