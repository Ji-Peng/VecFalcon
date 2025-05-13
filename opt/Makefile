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
# PG = -pg
# We work on the SHAKE256X4 variant
X4 = -DFNDSA_SHAKE256X4=1
LDFLAGS = $(PG)
LIBS = 
LIBS_GPERF = -Wl,-rpath -Wl,/usr/local/lib -lprofiler
# -MT $@: Set the target name in the dependency file to the current rule's target ($@).
# -MMD: Generate a dependency file, excluding system headers.
# -MP: Add empty phony targets for each dependency to prevent errors if headers are missing.
# -MF out/$*.d: Output the dependency file to the specified path (out/$*.d).
DEPFLAGS = -MT $@ -MMD -MP -MF out/$*.d

# We disable more advanced instruction sets to ensure a fair comparison. If we don't disable them explicitly, the compiler may convert SSE2 instructions to AVX instructions.
SSE2_FLAGS = -msse2 -mno-sse2avx -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-sse5 -mno-avx -mno-avx2 -mno-avx512f -mno-avx512dq -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx512vpopcntdq
AVX2_FLAGS = -mavx2 -mno-avx512f -mno-avx512dq -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx512vpopcntdq

CFLAGS_COMM = -W -Wextra -Wundef -Wshadow -O2 $(PG) $(X4) \
	-DWARMUP_N=2000 -DTESTS_N=10000 -g
CFLAGS_SSE2 = $(CFLAGS_COMM) -DFNDSA_AVX2=0 -DFNDSA_SSE2=1 $(SSE2_FLAGS)
CFLAGS_AVX2 = $(CFLAGS_COMM) -DFNDSA_AVX2=1 -DFNDSA_SSE2=1 $(AVX2_FLAGS)

SRCS_COMM = codec.c mq.c sha3.c sysrng.c util.c
SRCS_KGEN = kgen.c kgen_fxp.c kgen_gauss.c kgen_mp31.c kgen_ntru.c kgen_poly.c kgen_zint31.c
SRCS_SIGN = sign.c sign_core.c sign_fpoly.c sign_fpr.c sign_sampler.c
SRCS_VRFY = vrfy.c
SRCS = $(SRCS_COMM) $(SRCS_KGEN) $(SRCS_SIGN) $(SRCS_VRFY)
SRCS_TEST = test_fndsa.c
SRCS_SPEED = speed_fndsa.c
SRCS_GPERF = test_gperf.c

OBJS_SSE2		= $(addprefix out/, $(SRCS:.c=_sse2.o))
OBJS_TEST_SSE2	= $(addprefix out/, $(SRCS_TEST:.c=_sse2.o))
OBJS_SPEED_SSE2	= $(addprefix out/, $(SRCS_SPEED:.c=_sse2.o))
OBJS_GPERF_SSE2	= $(addprefix out/, $(SRCS_GPERF:.c=_sse2.o))

OBJS_AVX2		= $(addprefix out/, $(SRCS:.c=_avx2.o))
OBJS_TEST_AVX2	= $(addprefix out/, $(SRCS_TEST:.c=_avx2.o))
OBJS_SPEED_AVX2	= $(addprefix out/, $(SRCS_SPEED:.c=_avx2.o))
OBJS_GPERF_AVX2	= $(addprefix out/, $(SRCS_GPERF:.c=_avx2.o))

# for debug
SRCS_DEBUG = prof_sampler_next_sse2_2w.c

# all: \
# 	out/test_fndsa_sse2 out/speed_fndsa_sse2 out/test_gperf_sse2 \
# 	out/test_fndsa_avx2 out/speed_fndsa_avx2 out/test_gperf_avx2

all: \
	out/speed_fndsa_sse2 out/speed_fndsa_avx2 \
	out/test_fndsa_sse2  out/test_fndsa_avx2 out/prof_sampler_next_sse2_2w

run_speed: all
	echo "We work on the SHAKE256X4 variant.\n" >> speed_fndsa_11700k.txt
	echo "SSE2 version:" >> speed_fndsa_11700k.txt
	taskset -c 0 ./out/speed_fndsa_sse2 >> speed_fndsa_11700k.txt
	echo "AVX2 version:" >> speed_fndsa_11700k.txt
	taskset -c 0 ./out/speed_fndsa_avx2 >> speed_fndsa_11700k.txt

clean:
	-rm -rf out/

out:
	@mkdir -p out

out/prof_sampler_next_sse2_2w: $(SRCS_DEBUG) sha3.c ../profiling/cpucycles.c | out
	$(CC) $(CFLAGS_SSE2) $^ -o $@ $(LIBS)
	objdump -S $@ > $@_S.SS

out/test_fndsa_sse2: $(OBJS_SSE2) $(OBJS_TEST_SSE2) | out
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

out/speed_fndsa_sse2: $(OBJS_SSE2) $(OBJS_SPEED_SSE2) | out
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

# out/test_gperf_sse2: $(OBJS_SSE2) $(OBJS_GPERF_SSE2) | out
# 	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS_GPERF)

out/test_fndsa_avx2: $(OBJS_AVX2) $(OBJS_TEST_AVX2) | out
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

out/speed_fndsa_avx2: $(OBJS_AVX2) $(OBJS_SPEED_AVX2) | out
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

# out/test_gperf_avx2: $(OBJS_AVX2) $(OBJS_GPERF_AVX2) | out
# 	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS_GPERF)

out/%_sse2.o: %.c | out
	$(CC) $(CFLAGS_SSE2) $(DEPFLAGS) -c $< -o $@

out/%_avx2.o: %.c | out
	$(CC) $(CFLAGS_AVX2) $(DEPFLAGS) -c $< -o $@

-include $(wildcard out/*.d)
