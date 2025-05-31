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

For evaluating the practical limitations of the cpu we will initialize a vector `a` of size 8 (so the compiler can easily auto vectorize the inner for loop) and then mutiply it with a scalar `b` and add it back to itself. We will do this $M * N$ times. The main part of the code is very straight forward and simple we do $a[k] = a[k] \times b + a[k], \forall k\in [8]$ $MN$ times. This code does a total of $16MN + 16N$ flops, since M is of the order of `1e5` we will ignore the `16N`. This gives us a total of `16MN` flops
```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < M; i++) {
    float d[8] = {};
    for (int j = 0; j < N; j++)
        for (int k = 0; k < 8; k++)
            d[k] = a[k] * b + d[k];
    float s = 0;
    for (int k = 0; k < 8; k++) s += d[k];
    out[i] += s;
}
```

We time how long it takes to run this using `std::chrono::high_resolution_clock` and we will also measure the clock cycles during it's execution using `__rdtscp` intrinsics this will help us get the measured clock freq. We take an average of 5 runs for reporting the wall clock time and the cpu cycles. The entire code is at [cpu_flops.cpp](./cpu_flops.cpp) we will run this with aggressive auto vectorization and loop unrolling. [run_cpu.sh](./run_cpu.sh) is a small bash script which runs this code here are the flags that are enabled in it

```bash
export OMP_NUM_THREADS=16
g++ -fopenmp -O3 -march=native -ffast-math -funroll-loops max_flops.cpp -o main
./main
```

On running `source run_flops.sh` it prints this
```bash
Wall time            : 1.32544 s
Total FLOPs          : 1600 GFLOP
Achieved FLOPS       : 1207.15 GFLOP/s
Cycles elapsed       : 4529739898
Measured CPU freq    : 3.41754 GHz
Checksum             : 1.02817e+06
```
We also printed the first element of output after the exectution so the compiler does not optimize away everything. The main things to see is we achieve 1207.15 GFLOP/s and measured a clock freq of 3.41GHz. So our default clock freq matches from what was reported by the source. So we achieved 92.50% of the peak theoretical FLOPS