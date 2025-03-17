CC = gcc
COMMONDIR = ../common
CFLAGS = -W -Wextra -Wshadow -g \
	-O2 -mavx2 -mbmi2 -mpopcnt -march=native -mtune=native

SRC_COMM = cpucycles.c speed_print.c sha3.c

all: out/gaussian0 out/gaussian0_x4

clean:
	rm -rf out/

out/gaussian0: $(SRC_COMM) gaussian0.c
	mkdir -p out
	$(CC) $(CFLAGS) $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -D $@ > $@.SS

out/gaussian0_x4: $(SRC_COMM) gaussian0.c
	mkdir -p out
	$(CC) $(CFLAGS) -DFNDSA_AVX2 -DFNDSA_SHAKE256X4 $(filter %.c, $^) $(filter %.S, $^) -o $@
	objdump -D $@ > $@.SS