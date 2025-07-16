CC = gcc
CFLAGS_COMM = -W -Wextra -Wshadow -g -O2 -DFNDSA_SHAKE256X4=1
CFLAGS_AVX2 = $(CFLAGS_COMM) -mavx2 \
	-mno-avx512f -mno-avx512dq -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx512vpopcntdq
CFLAGS_SSE2 = $(CFLAGS_COMM) -msse2 \
	-mno-sse2avx -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-sse5 -mno-avx -mno-avx2 -mno-avx512f -mno-avx512dq -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx512vpopcntdq
CFLAGS_AVX512F = $(CFLAGS_COMM) -mavx512f \
	-mno-avx512dq -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx512vpopcntdq
# -MT $@: Set the target name in the dependency file to the current rule's target ($@).
# -MMD: Generate a dependency file, excluding system headers.
# -MP: Add empty phony targets for each dependency to prevent errors if headers are missing.
# -MF out/$*.d: Output the dependency file to the specified path (out/$*.d).
DEPFLAGS = -MT $@ -MMD -MP -MF $@.d

SRC_COMM = cpucycles.c sha3.c

all: \
	out/gaussian0_avx512f out/gaussian0_avx2 out/gaussian0_sse2 \
	out/batch_gaussian0_avx512f out/batch_gaussian0_avx2 out/batch_gaussian0_sse2 \
	out/batch_gaussian0_ref_avx512f out/batch_gaussian0_ref_avx2 out/batch_gaussian0_ref_sse2 \
	out/ber_exp_sse2

run_speed: all
	echo "We work on the SHAKE256X4 variant.\nbatch_gaussian0_ref with SHAKE256X4-AVX2 and gaussian0_ref:" > speed_batch_gaussian0_11700k.txt
	taskset -c 0 ./out/batch_gaussian0_ref_avx2 speed >> speed_batch_gaussian0_11700k.txt
	echo "\nbatch_gaussian0_sse2 with SHAKE256X2-SSE2 and gaussian0_sse2:" >> speed_batch_gaussian0_11700k.txt
	taskset -c 0 ./out/batch_gaussian0_sse2 speed >> speed_batch_gaussian0_11700k.txt
	echo "\nbatch_gaussian0_avx2 with SHAKE256X4-AVX2 and gaussian0_avx2:" >> speed_batch_gaussian0_11700k.txt
	taskset -c 0 ./out/batch_gaussian0_avx2 speed >> speed_batch_gaussian0_11700k.txt
	echo "\nbatch_gaussian0_avx512f with SHAKE256X8-AVX512 and gaussian0_avx512f:" >> speed_batch_gaussian0_11700k.txt
	taskset -c 0 ./out/batch_gaussian0_avx512f speed >> speed_batch_gaussian0_11700k.txt
	echo "We work on the SHAKE256X4 variant.\n" > speed_gaussian0_11700k.txt
	echo "SSE2 version:" >> speed_gaussian0_11700k.txt
	taskset -c 0 ./out/gaussian0_sse2 >> speed_gaussian0_11700k.txt
	echo "\nAVX2 version:" >> speed_gaussian0_11700k.txt
	taskset -c 0 ./out/gaussian0_avx2 >> speed_gaussian0_11700k.txt
	echo "\nAVX-512F with SHAKE256X8 version:" >> speed_gaussian0_11700k.txt
	taskset -c 0 ./out/gaussian0_avx512f >> speed_gaussian0_11700k.txt

run_test: all
	./out/batch_gaussian0_avx512f test > ./out/batch_gaussian0_avx512f.txt
	./out/batch_gaussian0_ref_avx512f test > ./out/batch_gaussian0_ref_avx512f.txt
	diff ./out/batch_gaussian0_avx512f.txt ./out/batch_gaussian0_ref_avx512f.txt
	./out/batch_gaussian0_avx2 test > ./out/batch_gaussian0_avx2.txt
	./out/batch_gaussian0_ref_avx2 test > ./out/batch_gaussian0_ref_avx2.txt
	diff ./out/batch_gaussian0_avx2.txt ./out/batch_gaussian0_ref_avx2.txt
	./out/batch_gaussian0_sse2 test > ./out/batch_gaussian0_sse2.txt
	./out/batch_gaussian0_ref_sse2 test > ./out/batch_gaussian0_ref_sse2.txt
	diff ./out/batch_gaussian0_sse2.txt ./out/batch_gaussian0_ref_sse2.txt

clean:
	rm -rf out/

out:
	@mkdir -p out

out/gaussian0_avx512f: $(SRC_COMM) gaussian0_avx512f.c KeccakP-1600-times8-SIMD512.c | out
	$(CC) $(CFLAGS_AVX512F) $(DEPFLAGS) -DFNDSA_SHAKE256X8=1 -DFNDSA_AVX512F=1 -DFNDSA_AVX2=1 -DFNDSA_SSE2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -S $@ > $@.SS

out/gaussian0_avx2: $(SRC_COMM) gaussian0_avx2.c | out
	$(CC) $(CFLAGS_AVX2) $(DEPFLAGS) -DFNDSA_AVX2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -S $@ > $@.SS

out/gaussian0_sse2: $(SRC_COMM) gaussian0_sse2.c | out
	$(CC) $(CFLAGS_SSE2) $(DEPFLAGS) -DFNDSA_AVX2=0 -DFNDSA_SSE2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -S $@ > $@.SS

out/batch_gaussian0_avx512f: $(SRC_COMM) test_batch_gaussian0.c KeccakP-1600-times8-SIMD512.c | out
	$(CC) $(CFLAGS_AVX512F) $(DEPFLAGS) -DFNDSA_SHAKE256X8=1 -DFNDSA_AVX512F=1 -DFNDSA_AVX2=1 -DFNDSA_SSE2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@

out/batch_gaussian0_avx2: $(SRC_COMM) test_batch_gaussian0.c | out
	$(CC) $(CFLAGS_AVX2) $(DEPFLAGS) -DFNDSA_AVX2=1 -DFNDSA_SSE2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@

out/batch_gaussian0_sse2: $(SRC_COMM) test_batch_gaussian0.c | out
	$(CC) $(CFLAGS_SSE2) $(DEPFLAGS) -DFNDSA_AVX2=0 -DFNDSA_SSE2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@

out/batch_gaussian0_ref_avx512f: $(SRC_COMM) test_batch_gaussian0.c | out
	$(CC) $(CFLAGS_AVX512F) $(DEPFLAGS) -DGAUSSIAN0_REF=1 -DFNDSA_SHAKE256X8=1 -DX8_REF=1 -DFNDSA_AVX512F=1 -DFNDSA_AVX2=0 -DFNDSA_SSE2=0 $(filter %.c, $^) $(filter %.S, $^) -o $@

out/batch_gaussian0_ref_avx2: $(SRC_COMM) test_batch_gaussian0.c | out
	$(CC) $(CFLAGS_AVX2) $(DEPFLAGS) -DGAUSSIAN0_REF=1 -DFNDSA_AVX2=1 -DFNDSA_SSE2=0 $(filter %.c, $^) $(filter %.S, $^) -o $@

out/batch_gaussian0_ref_sse2: $(SRC_COMM) test_batch_gaussian0.c | out
	$(CC) $(CFLAGS_SSE2) $(DEPFLAGS) -DGAUSSIAN0_REF=1 -DFNDSA_AVX2=0 -DFNDSA_SSE2=1 $(filter %.c, $^) $(filter %.S, $^) -o $@

out/ber_exp_sse2: $(SRC_COMM) ber_exp_sse2.c | out
	$(CC) $(CFLAGS_SSE2) $(DEPFLAGS) $(filter %.c, $^) $(filter %.S, $^) -o $@
	# objdump -S $@ > $@.SS

-include $(wildcard out/*.d)
