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

.if RVV == 1

FUNC init_vector_e16
    vsetivli a0, 16, e16, m1, tu, mu
ret

FUNC init_vector_e32
    vsetivli a0, 8, e32, m1, tu, mu
ret

FUNC init_vector_e64
    vsetivli a0, 4, e64, m1, tu, mu
ret

FUNC gaussian0_rvv_24b_p0_x2
.rep 100
    vmsbc.vx  v0, v20, t0       // low 24b 0th
    vmsbc.vx  v1, v21, t0       // low 24b 1th

    vmsbc.vxm v3, v20, t0, v0	// mid 24b 0th
    vmv.v.v   v0, v1
    vmsbc.vxm v0, v21, t0, v0	// mid 24b 1th

    vmsbc.vxm v0,v20, t0, v0	// high 24b  1th
    vadd.vx  v11, v11, t0, v0.t // vr1+=1 with mask
    vmv.v.v   v0, v3
    vmsbc.vxm v0,v21, t0, v0	// high 24b  0th
    vadd.vx  v10, v10, t0, v0.t // vr0+=1 with mask
.endr
FUNC_END gaussian0_rvv_24b_p0_x2

FUNC gaussian0_rvv_24b_p0_x3
.rep 100
    vmsbc.vx  v0, v20, t0       // low 24b 0th
    vmsbc.vx  v1, v21, t0       // low 24b 1th
    vmsbc.vx  v2, v22, t0       // low 24b 2th

    vmsbc.vxm v3, v20, t0, v0	// mid 24b 0th
    vmv.v.v   v0, v1
    vmsbc.vxm v1, v21, t0, v0	// mid 24b 1th
    vmv.v.v   v0, v2
    vmsbc.vxm v0, v22, t0, v0	// mid 24b 2th
    
    vmsbc.vxm v0,v22, t0, v0	// high 24b  2th
    vadd.vx v12, v12, t0, v0.t  // vr2+=1 with mask
    vmv.v.v   v0, v3
    vmsbc.vxm v0,v20, t0, v0	// high 24b  0th
    vadd.vx v10, v10, t0, v0.t  // vr0+=1 with mask
    vmv.v.v   v0, v1
    vmsbc.vxm v0,v21, t0, v0	// high 24b  1th
    vadd.vx v11, v11, t0, v0.t  // vr1+=1 with mask
.endr
FUNC_END gaussian0_rvv_24b_p0_x3

FUNC gaussian0_rvv_24b_p1_x3
.rep 100
    vmsbc.vx  v0, v20, t0       // low 24b 0th
    vsub.vx   v1, v21, t0       // low 24b 1th step 0
    vsub.vx   v2, v22, t0       // low 24b 2th step 0
    vsrl.vi   v1, v1, 31        // low 24b 1th step 1
    vsrl.vi   v2, v2, 31        // low 24b 2th step 1

    vmsbc.vxm v0, v20, t0, v0	// mid 24b 0th
    vsub.vx   v3, v21, t0       // mid 24b 1th step 0
    vsub.vx   v4, v22, t0       // mid 24b 2th step 0
    vsub.vv   v3, v3, v1        // mid 24b 1th step 1
    vsub.vv   v4, v4, v2        // mid 24b 2th step 1
    vsrl.vi   v3, v3, 31        // mid 24b 1th step 2
    vsrl.vi   v4, v4, 31        // mid 24b 2th step 2
    
    vmsbc.vxm v0,v20, t0, v0	// high 24b 0th
    vadd.vx v10, v10, t0, v0.t  // vr0+=1 with mask
    vsub.vx   v5, v21, t0       // high 24b 1th step 0
    vsub.vx   v6, v22, t0       // high 24b 2th step 0
    vsub.vv   v5, v5, v3        // high 24b 1th step 1
    vsub.vv   v6, v6, v4        // high 24b 2th step 1
    vsrl.vi   v5, v5, 31        // high 24b 1th step 2
    vsrl.vi   v6, v6, 31        // high 24b 2th step 2
    vadd.vv   v11,v11,v5        // vr1+=1
    vadd.vv   v12,v12,v6        // vr2+=1
.endr
FUNC_END gaussian0_rvv_24b_p1_x3

FUNC gaussian0_rvv_24b_p1_x3_v2
.rep 100
    vmsbc.vx  v0, v20, t0       // low 24b 0th
    vsub.vx   v1, v21, t0       // low 24b 1th step 0
    vsub.vx   v2, v22, t0       // low 24b 2th step 0
    vsub.vx   v3, v21, t0       // mid 24b 1th step 0
    vsub.vx   v4, v22, t0       // mid 24b 2th step 0
    vsrl.vi   v1, v1, 31        // low 24b 1th step 1
    vmsbc.vxm v0, v20, t0, v0	// mid 24b 0th
    vsrl.vi   v2, v2, 31        // low 24b 2th step 1
    vsub.vv   v3, v3, v1        // mid 24b 1th step 1
    vsub.vv   v4, v4, v2        // mid 24b 2th step 1

    vsub.vx   v5, v21, t0       // high 24b 1th step 0
    vsub.vx   v6, v22, t0       // high 24b 2th step 0
    vsrl.vi   v3, v3, 31        // mid 24b 1th step 2
    vmsbc.vxm v0,v20, t0, v0	// high 24b 0th
    vsrl.vi   v4, v4, 31        // mid 24b 2th step 2
    
    vsub.vv   v5, v5, v3        // high 24b 1th step 1
    vsub.vv   v6, v6, v4        // high 24b 2th step 1
    vsrl.vi   v5, v5, 31        // high 24b 1th step 2
    vadd.vx v10, v10, t0, v0.t  // vr0+=1 with mask
    vsrl.vi   v6, v6, 31        // high 24b 2th step 2
    vadd.vv   v11,v11,v5        // vr1+=1
    vadd.vv   v12,v12,v6        // vr2+=1
.endr
FUNC_END gaussian0_rvv_24b_p1_x3_v2

# 2w with vmsbc
FUNC gaussian0_rvv_64b_p0_x2
.rep 100
    vmsbc.vx  v0, v20, t0       // low 64b 0th
    vmsbc.vx  v1, v21, t0       // low 64b 1th
    vmsbc.vxm v0, v22, t0, v0   // high 8b 0th
    vadd.vx   v2, v2, t0, v0.t   // vr0+=1 with mask
    vmv.v.v   v0, v1
    vmsbc.vxm v0, v23, t0, v0   // high 8b 1th
    vadd.vx   v3, v3, t0, v0.t   // vr1+=1 with mask
.endr
FUNC_END gaussian0_rvv_64b_p0_x2

# 3w with vmsbc
FUNC gaussian0_rvv_64b_p0_x3
.rep 100
    vmsbc.vx  v0, v20, t0       // low 64b 0th
    vmsbc.vx  v1, v21, t0       // low 64b 1th
    vmsbc.vx  v2, v22, t0       // low 64b 2th

    vmsbc.vxm v0, v23, t0, v0   // high 8b 0th
    vadd.vx   v3, v3, t0, v0.t   // vr0+=1 with mask
    vmv.v.v   v0, v1
    vmsbc.vxm v0, v23, t0, v0   // high 8b 1th
    vadd.vx   v4, v4, t0, v0.t   // vr1+=1 with mask
    vmv.v.v   v0, v2
    vmsbc.vxm v0, v23, t0, v0   // high 8b 2th
    vadd.vx   v5, v5, t0, v0.t   // vr2+=1 with mask
.endr
FUNC_END gaussian0_rvv_64b_p0_x3

# 3w: 1w using vmsbc + 2w using 2xvsub+vsrl
FUNC gaussian0_rvv_64b_p1_x3
.rep 100
    vmsbc.vx  v0, v20, t0       // low 64b 0th
    vsub.vx   v1, v20, t0       // low 64b 1th step 0
    vsub.vx   v2, v20, t0       // low 64b 2th step 0
    vsrl.vi   v1, v1, 31        // low 64b 1th step 1
    vsrl.vi   v2, v2, 31        // low 64b 2th step 1

    vmsbc.vxm v0, v22, t0, v0   // high 8b 0th
    vadd.vx   v3, v3, t0, v0.t   // vr0+=1 with mask
    vsub.vx   v4, v22, t0       // high 8b 1th step 0
    vsub.vv   v4, v4, v1        // high 8b 1th step 1
    vsrl.vi   v4, v4, 31        // high 8b 1th step 2
    vadd.vv   v6, v6, v4        // vr1+=1
    vsub.vx   v5, v22, t0       // high 8b 2th step 0
    vsub.vv   v5, v5, v1        // high 8b 2th step 1
    vsrl.vi   v5, v5, 31        // high 8b 2th step 2
    vadd.vv   v7, v7, v5        // vr2+=1
.endr
FUNC_END gaussian0_rvv_64b_p1_x3

# 3w: 1w using vmsbc + 2w using 2xvsub+vsrl
FUNC gaussian0_rvv_64b_p1_x3_v2
.rep 100
    vsub.vx   v1, v20, t0       // low 64b 1th step 0
    vsub.vx   v2, v20, t0       // low 64b 2th step 0
    vmsbc.vx  v0, v20, t0       // low 64b 0th
    vsub.vx   v4, v22, t0       // high 8b 1th step 0
    vsrl.vi   v1, v1, 31        // low 64b 1th step 1
    vsub.vx   v5, v22, t0       // high 8b 2th step 0
    vsrl.vi   v2, v2, 31        // low 64b 2th step 1
    vmsbc.vxm v0, v22, t0, v0   // high 8b 0th
    vsub.vv   v4, v4, v1        // high 8b 1th step 1
    vsub.vv   v5, v5, v2        // high 8b 2th step 1
    vsrl.vi   v4, v4, 31        // high 8b 1th step 2
    vadd.vx   v3, v3, t0, v0.t   // vr0+=1 with mask
    vsrl.vi   v5, v5, 31        // high 8b 2th step 2
    vadd.vv   v6, v6, v4        // vr1+=1
    vadd.vv   v7, v7, v5        // vr2+=1
.endr
FUNC_END gaussian0_rvv_64b_p1_x3_v2

.endif