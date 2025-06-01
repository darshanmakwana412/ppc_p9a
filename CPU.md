## Theoretical limitations of CPU

For the cpu running `lscpu` gives
```bash
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   24
  On-line CPU(s) list:    0-23
Vendor ID:                GenuineIntel
  Model name:             13th Gen Intel(R) Core(TM) i7-13700K
    CPU family:           6
    Model:                183
    Thread(s) per core:   2
    Core(s) per socket:   16
    Socket(s):            1
    Stepping:             1
    CPU max MHz:          5400.0000
    CPU min MHz:          800.0000
    BogoMIPS:             6835.20
    Flags:                avx avx2 avx_vnni fma ...
Caches (sum of all):      
  L1d:                    640 KiB (16 instances)
  L1i:                    768 KiB (16 instances)
  L2:                     24 MiB (10 instances)
  L3:                     30 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-23
```
It shows that the cpu is 13th Gen Intel(R) Core(TM) i7-13700K and has 16 cores and 24 threads, looking more into the [intel doc](https://www.intel.com/content/www/us/en/products/sku/230500/intel-core-i713700k-processor-30m-cache-up-to-5-40-ghz/specifications.html) of i7-13700K it has 8 performant cores (with hyper threading of 2) and 8 efficient cores. For the P-cores it has a base frequency of 3.4GHz and turbo frequency of 5.3 GHz while for the E-cores it has a base frequency of 2.5GHz and turbo frequency of 4.2GHz

The i7-13700K operates at a default frequency of [3.4GHz](https://www.techpowerup.com/cpu-specs/core-i7-13700k.c2850). Since it supports avx2 and each p cores can do 2 inst/cycle on ports 0 and 1 we can achieve 2 * 8 (8 sp per 256bit vector) * 2 (2 flops per fma) = 32 flops/cycle. For 8 p cores we can achieve at max 8 * 32 * 5.3 GHz = 1356.8 GFLOP/s at turbo frequency

For each e core we achieve 16 flops/cycle and thus we can achieve at max 8 * 16 * 4.2 GHz = 537.6 GFLOP/s at turbo frequency

Total we can achieve a max flops of 1894.4 GFLOP/s

[1] - [13th Gen Intel (R) Core(TM) i7-13700K](https://www.intel.com/content/www/us/en/products/sku/230500/intel-core-i713700k-processor-30m-cache-up-to-5-40-ghz/specifications.html)

[2] - [Default clock freq of i7-13700K](https://www.techpowerup.com/cpu-specs/core-i7-13700k.c2850)

## Practical limitations of CPU

For evaluating the practical limitations of the cpu we will load an 8 element vector `a` into an avx256 register `a8` and broadcast a scalar `b` into another avx256 register `b8`. Then we perform fused multiply and add operation `d8[k] = a8 * b* + d8[k]` for each of the 8 lanes while repeating this `MN` times in parallel. This code does a total of $128MN + 64N$ flops, since M is of the order of `1e5` we will ignore the `64N`. This gives us a total of `128MN` flops. Here is the snippet of the main code
```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (uint64_t i = 0; i < M; i++) {

    __m256 d8[8] = {};
    for(int k=0; k<8; k++) {
        // randomly initialize d8[k]
        alignas(32) float v[8] = {};
        for (int j = 0; j < 8; j++) v[j] = rand() / float(RAND_MAX) * 2.0f - 1.0f;
        d8[k] = _mm256_load_ps(v);
    }
    for (uint64_t j = 0; j < N; j++) {
        #pragma unroll
        for(int k=0; k<8; k++) {
            d8[k] = _mm256_fmadd_ps(a8, b8, d8[k]);
        }
    }

    // Since N is much larger the flops of this is negligible
    __m256 s = _mm256_setzero_ps();
    for(int i=0; i<8; i++) {
        s = _mm256_add_ps(s, d8[i]);
    }
    out[i] += s[0];
}
```

We time how long it takes to run this using `std::chrono::high_resolution_clock` and we will also measure the clock cycles during it's execution using `__rdtscp` intrinsics this will help us get the measured clock freq. We take an average of 5 runs for reporting the wall clock time and the cpu cycles and thus the clock freq and the measured GSLOP/s. The entire code is at [cpu_flops.cpp](./cpu_flops.cpp), we will run this with the following flags enabled
```bash
export OMP_NUM_THREADS=24

g++ -fopenmp -g -O3 -march=native cpu_flops.cc -o main

perf stat ./main
```

Running it prints the following
```bash
Avg Wall time        : 3.50322 s
Total FLOPs          : 6400 GFLOP
Avg Achieved FLOP    : 1826.89 GFLOP/s
Cycles elapsed       : 11972383729
Measured CPU freq    : 3.41753 GHz
sum                  : 2.05752

 Performance counter stats for './main':

         80,781.54 msec task-clock                #   23.051 CPUs utilized          
             2,372      context-switches          #   29.363 /sec                   
               159      cpu-migrations            #    1.968 /sec                   
               393      page-faults               #    4.865 /sec                   
   389,859,205,718      cycles                    #    4.826 GHz                    
   500,717,583,488      instructions              #    1.28  insn per cycle         
    50,156,378,434      branches                  #  620.889 M/sec                  
         1,403,191      branch-misses             #    0.00% of all branches        

       3.504431292 seconds time elapsed

      80.759760000 seconds user
       0.023995000 seconds sys
```
We also print the sum of output after the exectution so the compiler does not optimize away everything. The main things to see is we achieve 1826.89 GFLOP/s and measured a clock freq of 4.826 GHz, there is also a huge difference between the clock frequencies measured by perf and __rdtscp. The reason could be __rdtscp does not vary according to the current core's frequency. Infact after reading more about it from [Time_Stamp_Counter](https://en.wikipedia.org/wiki/Time_Stamp_Counter) I came to realize it's use is highly discouraged. We achieved 96.43% of the theoretical peak flops. I think this is the practical best performance we can achieve as we never get the clock to run at turbo freq for all p and e core for all the avx256 instruction and there will be some dips in the clock freq degrading performance

Only running the [cpu_flops.cpp](./cpu_flops.cpp) on the p cores using
```bash
export OMP_NUM_THREADS=16
export OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export OMP_PROC_BIND=close

g++ -fopenmp -g -O3 -march=native cpu_flops.cc -o main

./main
```

gives us
```bash
Avg Wall time        : 4.79077 s
Total FLOPs          : 6400 GFLOP
Avg Achieved FLOP    : 1335.9 GFLOP/s
Cycles elapsed       : 16372601962
Measured CPU freq    : 3.41753 GHz
sum                  : 2.05752
```

Thus the p cores can reach 99.93% of their peak flops. while running it only on the e cores using
```bash
export OMP_NUM_THREADS=8
export OMP_PLACES="{16,17,18,19,20,21,22,23}"
export OMP_PROC_BIND=close

g++ -fopenmp -g -O3 -march=native cpu_flops.cc -o main

./main
```
gives us
```bash
Avg Wall time        : 11.9549 s
Total FLOPs          : 6400 GFLOP
Avg Achieved FLOP    : 535.346 GFLOP/s
Cycles elapsed       : 40856183393
Measured CPU freq    : 3.41753 GHz
sum                  : 2.05752
```
which is also 99.50% of the theoretical max that the e cores can achieve, I don't have any definite answers as to why only individually e cores and p cores are able to achieve max flops while together they fall short 2-3%

Looking at generated assembly code from [cpu_flops.s](./cpu_flops.s) we see that the compiler has generated the 8 fma256 instruction for us
```assembly
.L4:
	vfmadd231ps	%ymm0, %ymm1, %ymm2
	vfmadd231ps	%ymm0, %ymm1, %ymm9
	vfmadd231ps	%ymm0, %ymm1, %ymm8
	vfmadd231ps	%ymm0, %ymm1, %ymm7
	vfmadd231ps	%ymm0, %ymm1, %ymm6
	vfmadd231ps	%ymm0, %ymm1, %ymm5
	vfmadd231ps	%ymm0, %ymm1, %ymm4
	vfmadd231ps	%ymm0, %ymm1, %ymm3
	subq	$1, %rax
	jne	.L4
```

## Comparison with CP3B Solution

For comparison with my fastest cp3b solution, we use the exact same code as the [137707](https://ppc-exercises.cs.aalto.fi/course/aalto2025/cp/cp3b/137707) submission with replacing the avx512 instructions with avx256 and adding code for timing measurements and clock freq measurements. We set nx and ny to be 14000. On running
```
export OMP_NUM_THREADS=24

g++ -fopenmp -g -O3 -march=native cp3b.cc -o main

perf stat ./main
```
we see the following output
```
Wall time            : 1.73274 s
Total FLOPs          : 2744 GFLOP
Achieved FLOPS       : 1583.62 GFLOP/s
Cycles elapsed       : 5921721988
Measured CPU freq    : 3.41756 GHz
Checksum             : -0.00906668

 Performance counter stats for './main':

         45,399.47 msec task-clock                #   11.634 CPUs utilized          
               689      context-switches          #   15.176 /sec                   
                25      cpu-migrations            #    0.551 /sec                   
           683,659      page-faults               #   15.059 K/sec                  
   206,358,697,147      cycles                    #    4.545 GHz                    
   386,717,416,506      instructions              #    1.87  insn per cycle         
    15,785,912,832      branches                  #  347.711 M/sec                  
        62,951,893      branch-misses             #    0.40% of all branches        

       3.902154516 seconds time elapsed

      44.625873000 seconds user
       0.776241000 seconds sys
```
We achieved 83.59% of the theoretical peak flops

Now again inspecting the assembly at [cp3b.s](./cp3b.s) we see the compiler did the right thing the generated the vectorized code for us
```assembly
.L115:
	vmovaps	(%r8), %ymm1
	vbroadcastss	(%rdx), %ymm2
	addq	$32, %rdx
	vmovaps	(%r8,%rcx,4), %ymm0
	addq	$32, %r8
	vfmadd231ps	%ymm2, %ymm1, %ymm4
	vfmadd213ps	192(%rsp), %ymm0, %ymm2
	vmovaps	%ymm2, 192(%rsp)
	vbroadcastss	-28(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm9
	vfmadd213ps	128(%rsp), %ymm0, %ymm2
	vmovaps	%ymm2, 128(%rsp)
	vbroadcastss	-24(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm8
	vfmadd231ps	%ymm2, %ymm0, %ymm15
	vbroadcastss	-20(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm7
	vfmadd231ps	%ymm2, %ymm0, %ymm11
	vbroadcastss	-16(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm6
	vfmadd231ps	%ymm2, %ymm0, %ymm10
	vbroadcastss	-12(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm5
	vfmadd231ps	%ymm2, %ymm0, %ymm14
	vbroadcastss	-8(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm3
	vfmadd231ps	%ymm2, %ymm0, %ymm13
	vbroadcastss	-4(%rdx), %ymm2
	vfmadd213ps	160(%rsp), %ymm2, %ymm1
	vfmadd231ps	%ymm2, %ymm0, %ymm12
	vmovaps	%ymm1, 160(%rsp)
	cmpq	%rsi, %rdx
	jne	.L115
```