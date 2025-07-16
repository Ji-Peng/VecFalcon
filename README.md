# Vectorized Falcon-Sign

This is the artifact corresponding to the paper "Vectorized Falcon-Sign Implementations using SSE2, AVX2, AVX-512, NEON, and RVV".

## Directory Structure and Basic Project Organization

Directory structure:
- help/: some helper scripts
- opt/: optimized code implementations, supporting three target platforms
- profiling/: benchmarks for some subroutines, such as BaseSampler, FFT/iFFT
- ref/: reference implementation, derived from the public domain C-FN-DSA project (https://github.com/pornin/c-fn-dsa at commit id 96e3b92)

The three target platforms used in our paper:
- The Intel i7-11700K CPU (Rocket Lake microarchitecture) operating at 3.6 GHz. Hyper-Threading and Turbo Boost are disabled.
- The Cortex-A72 processor in Raspberry Pi 4B running at 1.5 GHz.
- The SpacemiT X60 core in Milk-V Jupiter operating at 2.0 GHz, supporting the RV64GCBV instruction set with vector extension v1.0 (VLEN = 256 bits) and bit-manipulation extension v1.0.0.

To precisely reproduce the performance data reported in our paper, ensure your hardware and software environment is as consistent as possible with our experimental environment (Section 2 of our paper).

Each of the directories opt/, profiling/, and ref/ contains three different Makefile files for compiling code for different platforms, namely `Makefile`, `Makefile.armv8a`, and `Makefile.rv`. You should specify the correct Makefile for compilation.
For example, in the profiling/ directory:
- On the Intel i7-11700K, you can compile using: `make all -j`
- On the ARM Cortex-A72, you can compile using: `make all -j -f Makefile.armv8a`
- On the SpacemiT X60, you can compile using: `make all -j -f Makefile.rv`

## Reproducing the Results in the Paper

### Table 2

To reproduce "Table 2: The performance profiling of Falcon-1024’s signature generation", you first need to install `gperftools`, which can be done with the following commands:

```bash
git clone https://github.com/gperftools/gperftools.git
cd gperftools
./autogen.sh
./configure
make
sudo make install
```

After running the above commands, the libraries will be installed in `/usr/local/lib`.
Then install the `pprof` tool. We recommend setting up a Go language environment first, then installing `pprof` via `go install github.com/google/pprof@latest`, and ensure its path is added to your PATH.

For the Intel i7-11700K, go to the ref/ directory: 
```bash
make all -j
make run_profiling
```

You will then get multiple txt files, such as `gperf_sign_core_1024_avx2.txt`, which contain the profiling results of the AVX2 version in Table 2.

For the SpacemiT X60, the steps are similar: first install `gperftools` and `pprof`, then run `make run_profiling -f Makefile.rv` in the ref/ directory.

### Table 3 and Table 4

To reproduce "Table 3: Benchmark results of various BaseSampler implementations" and "Table 4: Benchmark results of FFT/iFFT implementations on SpacemiT X60.", the main work is in the profiling/ directory.

For the Intel i7-11700K:
```bash
make all -j
make run_speed
```

You will then get `speed_gaussian0_11700k.txt`, which contains the experimental results of BaseSampler for SSE2, AVX2, and AVX-512F instruction sets in Table 2.

If you want to test the correctness of our BaseSampler implementation, run `make run_test`. If no output from the `diff` command is observed, it indicates that the test passed.

For the Cortex-A72:
```bash
make all -j -f Makefile.armv8a
make run_speed -f Makefile.armv8a
```

The file `speed_gaussian0_cortex_a72.txt` you get contains the experimental results of BaseSampler for the NEON instruction set in Table 2.

For the SpacemiT X60:
```bash
make all -j -f Makefile.rv
make run_speed -f Makefile.rv
```

The file `speed_gaussian0_x60.txt` you get contains the experimental results of BaseSampler for the RISC-V instruction set in Table 2.
The file `speed_fft_rv64d_x60.txt` you get contains the experimental results of FFT/iFFT for the RISC-V instruction set in Table 3.

### Table 5

To reproduce "Table 5: Benchmark results of Falcon-{512,1024}’s signature generation (sign_core subroutine) on three target platforms (8 distinct instruction set configurations).", the main work is in the ref/ and opt/ directories.

First, reproduce the results of the reference implementations in the ref/ directory for Table 5.

For the Intel i7-11700K:
```bash
make all -j
make run_speed
```
You will then get `speed_fndsa_11700k.txt`, which contains the experimental results of the reference implementations for the sign_core subroutine for SSE2, AVX2, and AVX-512F instruction sets in Table 5.

For the Cortex-A72:
```bash
make all -j -f Makefile.armv8a
make run_speed -f Makefile.armv8a
```

The file `speed_fndsa_cortex_a72.txt` you get contains the experimental results of the reference implementations for the sign_core subroutine for the NEON instruction set in Table 5.

For the SpacemiT X60:
```bash
make all -j -f Makefile.rv
make run_speed -f Makefile.rv
```

The file `speed_fndsa_x60.txt` you get contains the experimental results of the reference implementations for the sign_core subroutine for the RISC-V instruction set in Table 5.

Then, reproduce the results of our optimized implementations in the opt/ directory. The commands and the filenames of the files you get are the same as those in the ref/ directory, so they are not repeated here.

### Others

In Section 7, we mentioned: "For implementations using NEON, the performance improvement is 17% compared to the reference implementation. If we exclude the 4-way hybrid Keccak and optimized FFT/iFFT, the improvement reduces to 9%. Integrating our BaseSampler with the 4-way hybrid Keccak results in a 13% improvement over the reference implementation."

If you want to reproduce the result on Cortex-A72 for "If we exclude the 4-way hybrid Keccak and optimized FFT/iFFT, the improvement reduces to 9%":
In the opt/ directory, change `Makefile.armv8a` to ` -DFNDSA_NEON_HYBRID_SHA3=0 -DFNDSA_NEON_FFT_OPT=0`, then:

```bash
make clean -f Makefile.armv8a
make all -j -f Makefile.armv8a
make run_speed -f Makefile.armv8a
```


If you want to reproduce the result on Cortex-A72 for "Integrating our BaseSampler with the 4-way hybrid Keccak results in a 13% improvement over the reference implementation":
In the opt/ directory, change `Makefile.armv8a` to ` -DFNDSA_NEON_HYBRID_SHA3=1 -DFNDSA_NEON_FFT_OPT=0`, then:

```bash
make clean -f Makefile.armv8a
make all -j -f Makefile.armv8a
make run_speed -f Makefile.armv8a
```

In Section 7, we mentioned: "All four versions on RISC-V show significant improvements. ... Without the optimized Keccak, the improvement is 41% compared to the reference implementation."

If you want to reproduce the above result on SpacemiT X60:
In the opt/ directory, change `Makefile.rv` to `-DKECCAK_OPT=0`, then:

```bash
make clean -f Makefile.rv
make all -j -f Makefile.rv
make run_speed -f Makefile.rv
```

Section 7 mentions "our implementation using AVX2 increases the code size by approximately 2.7 KB compared to the reference implementation"
To reproduce this result, run the following commands in the ref/ and opt/ directories respectively, and then compare the results:

```bash
nm out/speed_fndsa_avx2 --print-size --size-sort --radix=d | \
awk '{$1=""}1' | \
awk '{sum+=$1 ; print $0} END{print "Total size =", sum, "bytes =", sum/1024, "kB"}' > speed_fndsa_avx2_symbols_size.txt
```
