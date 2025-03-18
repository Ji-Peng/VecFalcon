CC = gcc
COMMONDIR = ../common
CFLAGS = -W -Wextra -Wshadow -g \
	-O2 -mavx2 -mbmi2 -mpopcnt -march=native -mtune=native
# -MT $@: Set the target name in the dependency file to the current rule's target ($@).
# -MMD: Generate a dependency file, excluding system headers.
# -MP: Add empty phony targets for each dependency to prevent errors if headers are missing.
# -MF out/$*.d: Output the dependency file to the specified path (out/$*.d).
DEPFLAGS = -MT $@ -MMD -MP -MF $@.d

SRC_COMM = cpucycles.c sha3.c

all: out/gaussian0 out/gaussian0_x4

clean:
	rm -rf out/

out/gaussian0: $(SRC_COMM) gaussian0.c
	@mkdir -p out
	$(CC) $(CFLAGS) $(DEPFLAGS) $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -D $@ > $@.SS

out/gaussian0_x4: $(SRC_COMM) gaussian0.c
	@mkdir -p out
	$(CC) $(CFLAGS) $(DEPFLAGS) -DFNDSA_AVX2 -DFNDSA_SHAKE256X4 $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -D $@ > $@.SS

-include $(wildcard out/*.d)
