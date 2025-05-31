## Theoretical limitations

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

The i7-13700K operates at a default frequency of [3.4GHz](https://www.techpowerup.com/cpu-specs/core-i7-13700k.c2850). Since it supports avx2 and each p cores can do 2 inst/cycle on ports 0 and 1 we can achieve 2 * 8 (8 sp per 256bit vector) * 2 (2 flops per fma) = 32 flops/cycle. For 8 p cores we can achieve at max 8 * 32 * 3.4 GHz = 870.4 GFLOP/s

For each e core we achieve 16 flops/cycle and thus we can achieve at max 8 * 16 * 3.4 GHz = 435.2 GFLOP/s

Total we can achieve a max flops of 1305.6 GFLOP/s

[1] - [13th Gen Intel (R) Core(TM) i7-13700K](https://www.intel.com/content/www/us/en/products/sku/230500/intel-core-i713700k-processor-30m-cache-up-to-5-40-ghz/specifications.html)

[2] - [Default clock freq of i7-13700K](https://www.techpowerup.com/cpu-specs/core-i7-13700k.c2850)

## Practical limitations of CPU

For evaluating the practical limitations of the cpu we will initialize a vector `a` of size 8 into an avx256 rigister and then mutiply it with a scalar `b` broadcasted to also avx256 and add it back to itself. We will do this many times in parallel and in loop. The main part of the code is very straight forward and simple we do $a[k] = a[k] \times b + a[k], \forall k\in [8]$ $8MN$ times. This code does a total of $64MN + 64N$ flops, since M is of the order of `1e5` we will ignore the `64N`. This gives us a total of `64MN` flops. Here is the snippet of the code
```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < M; i++) {

    __m256 d8[8] = {};
    for (int j = 0; j < N; j++) {
        #pragma unroll
        for(int i=0; i<8; i++) {
            d8[i] = _mm256_fmadd_ps(a8, b8, d8[i]);
        }
    }

    // Since N is much larger the flops of this is negligible
    __m256 s = _mm256_setzero_ps();
    for(int i=0; i<8; i++) {
        s = _mm256_add_ps(s, d8[i]);
    }
    out[i] = s[0];
}
```

We time how long it takes to run this using `std::chrono::high_resolution_clock` and we will also measure the clock cycles during it's execution using `__rdtscp` intrinsics this will help us get the measured clock freq. We take an average of 5 runs for reporting the wall clock time and the cpu cycles. The entire code is at [cpu_flops.cpp](./cpu_flops.cpp) we will run this with aggressive auto vectorization and loop unrolling. [run_cpu.sh](./run_cpu.sh) is a small bash script which runs this code here are the flags that are enabled in it

```bash
Wall time            : 3.84917 s
Total FLOPs          : 6400 GFLOP
Achieved FLOPS       : 1662.7 GFLOP/s
Cycles elapsed       : 13154666166
Measured CPU freq    : 3.41754 GHz
Checksum             : 2.74405e+06

 Performance counter stats for './main':

         90,034.56 msec task-clock                #   23.383 CPUs utilized          
             1,332      context-switches          #   14.794 /sec                   
                40      cpu-migrations            #    0.444 /sec                   
               291      page-faults               #    3.232 /sec                   
   443,002,046,830      cycles                    #    4.920 GHz                    
   300,142,917,222      instructions              #    0.68  insn per cycle         
   100,030,393,458      branches                  #    1.111 G/sec                  
           382,083      branch-misses             #    0.00% of all branches        

       3.850457958 seconds time elapsed

      90.032370000 seconds user
       0.003998000 seconds sys
```
We also printed the first element of output after the exectution so the compiler does not optimize away everything. The main things to see is we achieve 1688.23 GFLOP/s and measured a clock freq of 3.41GHz. So our default clock freq matches from what was reported by the source. So we achieved 92.50% of the peak theoretical FLOPS

Only running the [cpu_flops.cpp](./cpu_flops.cpp) on the p cores using
```bash
export OMP_NUM_THREADS=16
export OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export OMP_PROC_BIND=close

g++ -fopenmp -g -O3 -march=native -ffast-math -funroll-loops cpu_flops.cpp -o main

./main
```

gives us
```bash
Wall time            : 1.8551 s
Total FLOPs          : 1600 GFLOP
Achieved FLOPS       : 862.489 GFLOP/s
Cycles elapsed       : 6339862401
Measured CPU freq    : 3.41754 GHz
Checksum             : 1.02817e+06
```
which is 99.08% of the theoretical max of p cores, while running the same for only e cores using
```bash
export OMP_NUM_THREADS=8
export OMP_PLACES="{16,17,18,19,20,21,22,23}"
export OMP_PROC_BIND=close

g++ -fopenmp -g -O3 -march=native -ffast-math -funroll-loops cpu_flops.cpp -o main

./main
```
gives
```bash
Wall time            : 4.47915 s
Total FLOPs          : 1600 GFLOP
Achieved FLOPS       : 357.211 GFLOP/s
Cycles elapsed       : 15307646282
Measured CPU freq    : 3.41753 GHz
Checksum             : 1.02817e+06
```
which is only 82% of the theoretical max that the e cores can achieve, I don't have any definite answers as to why only e cores are underperforming but I could be that they are more optimized for energy efficient compared to performance

## Comparison with CP3B Solution

For comparison with my fastest cp3b solution, we use the exact same code as the [137707](https://ppc-exercises.cs.aalto.fi/course/aalto2025/cp/cp3b/137707) submission with replacing the avx512 instructions with avx256 and adding code for timing measurements and clock freq measurements. We set nx and ny to be 14000. On running
```
export OMP_NUM_THREADS=24

g++ -fopenmp -g -O3 -march=native -ffast-math -funroll-loops cp3b.cc -o main

./main
```
we see the following output
```
Wall time            : 2.30501 s
Total FLOPs          : 2744 GFLOP
Achieved FLOPS       : 1190.45 GFLOP/s
Cycles elapsed       : 7877479611
Measured CPU freq    : 3.41754 GHz
Checksum             : -0.00906668
```
We achieved 91.16% of the theoretical maximum