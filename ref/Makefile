# Build script for the Falcon implementation.
#
# ==========================(LICENSE BEGIN)============================
#
# Copyright (c) 2017-2019  Falcon Project
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ===========================(LICENSE END)=============================
#
# @author   Thomas Pornin <thomas.pornin@nccgroup.com>

.POSIX:

# =====================================================================
#
# Configurable options:
#   CC       C compiler; GCC or Clang are fine; MSVC (2015+) works too.
#   CFLAGS   Compilation flags:
#             * Optimization level -O2 or higher is recommended
#            See config.h for some possible configuration macros.
#   LD       Linker; normally the same command as the compiler.
#   LDFLAGS  Linker options, not counting the extra libs.
#   LIBS     Extra libraries for linking:
#             * If using the native FPU, test_falcon and application
#               code that calls this library may need: -lm
#               (normally not needed on x86, both 32-bit and 64-bit)

COMMONDIR 		= ../common

CC = gcc
CFLAGS = -Wall -Wextra -Wshadow -Wundef -O3 -I$(COMMONDIR) #-pg -fno-pie
LD = gcc
LDFLAGS = #-pg -no-pie
LIBS = -lm

# =====================================================================

OBJ = codec.o common.o falcon.o fft.o fpr.o keygen.o rng.o shake.o sign.o vrfy.o cpucycles.o speed_print.o

all: test_vectors_falcon test_speed_falcon

clean:
	-rm -f $(OBJ) test_vectors_falcon test_speed_falcon test_vectors_falcon.o test_speed_falcon.o

test_vectors_falcon: test_vectors_falcon.o $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $(filter %.o, $^) $(LIBS)

test_speed_falcon: test_speed_falcon.o $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $(filter %.o, $^) $(LIBS)

codec.o: codec.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o codec.o codec.c

common.o: common.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o common.o common.c

falcon.o: falcon.c falcon.h config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o falcon.o falcon.c

fft.o: fft.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o fft.o fft.c

fpr.o: fpr.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o fpr.o fpr.c

keygen.o: keygen.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o keygen.o keygen.c

rng.o: rng.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o rng.o rng.c

shake.o: shake.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o shake.o shake.c

sign.o: sign.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o sign.o sign.c

vrfy.o: vrfy.c config.h inner.h fpr.h
	$(CC) $(CFLAGS) -c -o vrfy.o vrfy.c

cpucycles.o: $(COMMONDIR)/cpucycles.c $(COMMONDIR)/cpucycles.h
	$(CC) $(CFLAGS) -c -o $@ $(filter %.c, $^)

speed_print.o: $(COMMONDIR)/speed_print.c $(COMMONDIR)/speed_print.h $(COMMONDIR)/cpucycles.h
	$(CC) $(CFLAGS) -c -o $@ $(filter %.c, $^)

test_speed_falcon.o: test_speed_falcon.c falcon.h $(COMMONDIR)/speed_print.h $(COMMONDIR)/cpucycles.h
	$(CC) $(CFLAGS) -c -o test_speed_falcon.o test_speed_falcon.c

test_vectors_falcon.o: test_vectors_falcon.c falcon.h config.h inner.h fpr.h $(COMMONDIR)/speed_print.h $(COMMONDIR)/cpucycles.h
	$(CC) $(CFLAGS) -c -o test_vectors_falcon.o test_vectors_falcon.c
