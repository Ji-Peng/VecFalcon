# Possible options:
#   -DFNDSA_AVX2=0         disable AVX2 support
#   -DFNDSA_SSE2=0         disable SSE2 support
#   -DFNDSA_NEON=0         disable NEON support
#   -DFNDSA_RV64D=0        disable use of floating-point hardware on RISC-V
#
#   -DFNDSA_NEON_SHA3=1    enable NEON optimizations for parallel SHAKE256
#   -DFNDSA_DIV_EMU=1      force integer emulation of divisions (RISC-V only)
#   -DFNDSA_SQRT_EMU=1     force integer emulation of square roots (RISC-V only)
#
#   -DFNDSA_SHAKE256X4=1   use four parallel SHAKE256 as internal PRNG
#
# AVX2 support is compiled on x86 and x86_64 but is gated at runtime
# with a check that AVX2 is supported by the current CPU (and not
# disabled by the operating system); if AVX2 cannot be used, then the
# fallback code (normally with SSE2) is used. Thus, support of AVX2 does
# not prevent the code from running on non-AVX2 machines.
#
# SSE2 intrinsics are used if supported by the target architecture at
# compile-time (no runtime test); this is normally the case for 64-bit
# builds, since SSE2 is part of the 64-bit ABI. On 32-bit builds, this
# depends on the compiler's defaults, which are OS dependent (e.g. MSVC
# on Windows enables SSE2 by default, and so does Clang on 32-bit MacOS,
# but on 32-bit Linux this is not done by default, and you have to add
# the '-msse2' flag to the compiler to get SSE2). You can force SSE2
# usage even if not detected by setting '-DFNDSA_SSE2=1'.
#
# NEON intrisics are used if supported by the target architecture at
# compile-time (no runtime test) _and_ the build is 64-bit. Since NEON is
# part of the 64-bit ARMv8 ABI, you normally don't have to fiddle with
# that.
#
# An optional NEON-optimized SHAKE256 implementation can be enabled (it
# runs two SHAKE256 implementations in parallel). It is disabled by default
# because it turns out to be slower than the plain code on ARM Cortex-A55
# and Cortex-A76 test systems. It _might_ be faster on some other ARM
# systems. To enable it, use '-DFNDSA_NEON_SHA3=1'.
#
# On 64-bit RISC-V systems, the floating-point hardware is used if
# detected at compile-time (i.e. the target architecture includes the 'D'
# extension, which is part of the usual "RV64GC" package). When these
# instructions are used, the divisions and square roots may optionally
# be done with only integer computations, which is slower but possibly
# safer with regard to timing attacks.
#
# An internal PRNG is used during key pair generation (to generate
# candidate (f,g) polynomial pairs) and during signature generation (to
# power the Gaussian sampling). By default, that PRNG is a simple
# SHAKE256. An alternate PRNG is enabled by setting
# '-DFNDSA_SHAKE256X4=1': this new PRNG uses four SHAKE256 in parallel,
# with interleaved outputs. The alternate PRNG speeds up signature
# generation by about 20% when runing on an x86 CPU with AVX2 support;
# however, it also increases stack usage by aout 1.1 kB, which can be a
# problem on small embedded systems such as microcontrollers, which is
# why it is not the default. Moreover, using the alternate PRNG
# necessarily changes the keys and signatures obtained from a given seed
# (note that reproducibility of keys and signatures should not be relied
# upon, at least until the FN-DSA standard is finalized, as things are
# expected to change again in some areas).
#
# By default, this code compiles 'test_fndsa' (a test framework to validate
# that all computations are correct) and 'speed_fndsa' (speed benchmarks).

CC = gcc
LD = gcc
LDFLAGS = #-pg
LIBS =
DEPDIR = .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

CFLAGS_COMMON = -W -Wextra -Wundef -Wshadow -O2 #-pg
CFLAGS_RV64GC	 = $(CFLAGS_COMMON) -mabi=lp64d -march=rv64gc	\
	-DFNDSA_RV64D=1 -DRV64=1
CFLAGS_RV64GCV	 = $(CFLAGS_COMMON) -mabi=lp64d -march=rv64gcv	\
	-DFNDSA_RV64D=1 -DRV64=1 -DRVV=1
CFLAGS_RV64GCB	 = $(CFLAGS_COMMON) -mabi=lp64d -march=rv64gc_zba_zbb 	\
	-DFNDSA_RV64D=1 -DRV64=1 -DRVB=1
CFLAGS_RV64GCVB	 = $(CFLAGS_COMMON) -mabi=lp64d -march=rv64gcv_zba_zbb	\
	-DFNDSA_RV64D=1 -DRV64=1 -DRVV=1 -DRVB=1
	
SRCS_COMM = codec.c mq.c sha3.c sysrng.c util.c
SRCS_KGEN = kgen.c kgen_fxp.c kgen_gauss.c kgen_mp31.c kgen_ntru.c kgen_poly.c kgen_zint31.c
SRCS_SIGN = sign.c sign_core.c sign_fpoly.c sign_fpr.c sign_sampler.c
SRCS_VRFY = vrfy.c
SRCS = $(SRCS_COMM) $(SRCS_KGEN) $(SRCS_SIGN) $(SRCS_VRFY)
SRCS_TEST = test_fndsa.c test_sampler.c test_sign.c
SRCS_SPEED = speed_fndsa.c

ASMS_RV64IM 	= fips202_rv64im.S
ASMS_RV64IMB	= fips202_rv64imb.S

OBJS_RV64GC 		= $(addprefix out/, $(SRCS:.c=_rv64gc.o))	\
	$(addprefix out/, $(ASMS_RV64IM:.S=.o))
OBJS_RV64GCV 		= $(addprefix out/, $(SRCS:.c=_rv64gcv.o))	\
	$(addprefix out/, $(ASMS_RV64IM:.S=.o))
OBJS_RV64GCB 		= $(addprefix out/, $(SRCS:.c=_rv64gcb.o))	\
	$(addprefix out/, $(ASMS_RV64IMB:.S=.o))
OBJS_RV64GCVB 		= $(addprefix out/, $(SRCS:.c=_rv64gcvb.o))	\
	$(addprefix out/, $(ASMS_RV64IMB:.S=.o))
OBJS_TEST_RV64GC 	= $(addprefix out/, $(SRCS_TEST:.c=_rv64gc.o))
OBJS_TEST_RV64GCV 	= $(addprefix out/, $(SRCS_TEST:.c=_rv64gcv.o))
OBJS_TEST_RV64GCB 	= $(addprefix out/, $(SRCS_TEST:.c=_rv64gcb.o))
OBJS_TEST_RV64GCVB 	= $(addprefix out/, $(SRCS_TEST:.c=_rv64gcvb.o))
OBJS_SPEED_RV64GC 	= $(addprefix out/, $(SRCS_SPEED:.c=_rv64gc.o))
OBJS_SPEED_RV64GCV 	= $(addprefix out/, $(SRCS_SPEED:.c=_rv64gcv.o))
OBJS_SPEED_RV64GCB 	= $(addprefix out/, $(SRCS_SPEED:.c=_rv64gcb.o))
OBJS_SPEED_RV64GCVB = $(addprefix out/, $(SRCS_SPEED:.c=_rv64gcvb.o))

all: \
	out/test_fndsa_rv64gc 		out/speed_fndsa_rv64gc		\
	out/test_fndsa_rv64gcv 		out/speed_fndsa_rv64gcv		\
	out/test_fndsa_rv64gcb 		out/speed_fndsa_rv64gcb		\
	out/test_fndsa_rv64gcvb 	out/speed_fndsa_rv64gcvb

run_speed: \
	out/speed_fndsa_rv64gc 	out/speed_fndsa_rv64gcv		\
	out/speed_fndsa_rv64gcb	out/speed_fndsa_rv64gcvb
	echo "FN-DSA on RV64GC" >speed_fndsa_x60.txt
	perf stat ./out/speed_fndsa_rv64gc   >>speed_fndsa_x60.txt   2>/dev/null
	echo "FN-DSA on RV64GCV" >>speed_fndsa_x60.txt
	perf stat ./out/speed_fndsa_rv64gcv  >>speed_fndsa_x60.txt  2>/dev/null
	echo "FN-DSA on RV64GCB" >>speed_fndsa_x60.txt
	perf stat ./out/speed_fndsa_rv64gcb  >>speed_fndsa_x60.txt  2>/dev/null
	echo "FN-DSA on RV64GCVB" >>speed_fndsa_x60.txt
	perf stat ./out/speed_fndsa_rv64gcvb >>speed_fndsa_x60.txt 2>/dev/null

clean:
	-rm -rf out/

out/test_fndsa_rv64gc: $(OBJS_RV64GC) $(OBJS_TEST_RV64GC)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/speed_fndsa_rv64gc: $(OBJS_RV64GC) $(OBJS_SPEED_RV64GC)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/test_fndsa_rv64gcv: $(OBJS_RV64GCV) $(OBJS_TEST_RV64GCV)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/speed_fndsa_rv64gcv: $(OBJS_RV64GCV) $(OBJS_SPEED_RV64GCV)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/test_fndsa_rv64gcb: $(OBJS_RV64GCB) $(OBJS_TEST_RV64GCB)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/speed_fndsa_rv64gcb: $(OBJS_RV64GCB) $(OBJS_SPEED_RV64GCB)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/test_fndsa_rv64gcvb: $(OBJS_RV64GCVB) $(OBJS_TEST_RV64GCVB)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

out/speed_fndsa_rv64gcvb: $(OBJS_RV64GCVB) $(OBJS_SPEED_RV64GCVB)
	$(LD) $(LDFLAGS) $(LIBS) $^ -o $@

# -MMD option is used to analyze file dependencies
out/%_rv64gc.o: %.c
	mkdir -p out
	$(CC) $(CFLAGS_RV64GC) -MMD -MF out/$*.d -c $< -o $@

out/%_rv64gcv.o: %.c
	mkdir -p out
	$(CC) $(CFLAGS_RV64GCV) -MMD -MF out/$*.d -c $< -o $@

out/%_rv64gcb.o: %.c
	mkdir -p out
	$(CC) $(CFLAGS_RV64GCB) -MMD -MF out/$*.d -c $< -o $@

out/%_rv64gcvb.o: %.c
	mkdir -p out
	$(CC) $(CFLAGS_RV64GCVB) -MMD -MF out/$*.d -c $< -o $@

-include $(wildcard out/*.d)

out/%.o: %.S
	mkdir -p out
	$(CC) $(CFLAGS_RV64GCVB) -c $< -o $@
