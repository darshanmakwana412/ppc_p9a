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

Since it supports avx2 and each p cores can retire 2 inst/cycle on ports 0 and 1 we can achieve 2 * 8 (8 sp per 256bit vector) * 2 (2 flops per fma) = 32 flops/cycle. For 8 p cores we can achieve at max 8 * 32 * 5.4 GHz = 1356.8 GFLOPS

For each e core we achieve 16 flops/cycle and thus we can achieve at max 8 * 16 * 4.2 GHz = 537.6 GFLOPS

## Practical limitations of CPU

For this we will initialize a vector `a` of size 8 (so the compiler can easily auto vectorize the inner for loop) and then mutiply it with a scalar `b` and add it back to itself. We will do this $M * N$ times. The main part of the code is very straight forward and simple we do $a[k] = a[k] \times b + a[k], \forall k\in [8]$. This code does a total of $16MN + 16N$ flops, since M is of the order of `1e5` we will ignore the `16N`. This gives us a total of `16MN` flops
```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < M; i++) {
    float d[8] = {};
    for (int j = 0; j < N; j++)
        for (int k = 0; k < 8; k++)
            d[k] = a[k] * b + d[k];

    for (int k = 0; k < 8; k++) a[k] += d[k];
}
```

We time how long it takes to run this using `std::chrono::high_resolution_clock` and we will also measure the clock cycles during it's execution using `__rdtscp` intrinsics this will help us get the measured clock freq. The entire code is at [cpu_flops.cpp](./cpu_flops.cpp) we will run this with aggressive auto vectorization and loop unrolling. [run_cpu.sh](./run_cpu.sh) is a small bash script which runs this code here are the flags that are enabled in it. We also use perf stat for analyzing the instructions and branches and misses

```bash
export OMP_NUM_THREADS=16
g++ -fopenmp -O3 -march=native -ffast-math -funroll-loops max_flops.cpp -o main
perf stat ./main
```

On running `source run_flops.sh` it prints this
```bash
Wall time            : 1.40042 s
Total FLOPs          : 1600 GFLOP
Achieved FLOPS       : 1142.51 GFLOP/s
Cycles elapsed       : 4785974826
Measured CPU freq    : 3.41753 GHz
Checksum             : -nan

 Performance counter stats for './main':

        162,581.01 msec task-clock                #   23.215 CPUs utilized          
             2,183      context-switches          #   13.427 /sec                   
                72      cpu-migrations            #    0.443 /sec                   
               192      page-faults               #    1.181 /sec                   
   747,774,532,413      cycles                    #    4.599 GHz                    
 1,000,480,605,400      instructions              #    1.34  insn per cycle         
    31,334,714,086      branches                  #  192.733 M/sec                  
         1,157,529      branch-misses             #    0.00% of all branches        

       7.003167863 seconds time elapsed

     162.571710000 seconds user
       0.011986000 seconds sys
```
yeah we also print the sum of the vector `a` after the exectution so the compiler does does not optimize away everything it's inf because we elements of `a` unbounded as we keep on multiplying and adding it's not a big deal. The main things to see is we achieve 512.26 GFLOP/s and measured a clock freq of 2.11GHz. So our estimate of default clock freq was slightly underestimated when we recaculate the theoretical flops of single precision using 2.1GHz clock freq we get 537.6GFLOP/s. So we achieved 95.28% of the peak FLOPS

Inspecting the assembly we see the compiler optimized our inner most for loop
```s
.LBE136:
    .loc 1 26 9 discriminator 2 view .LVU94
    .loc 1 26 32 discriminator 1 view .LVU95
    vaddps	%ymm10, %ymm4, %ymm4
    vmovaps	608(%rsp), %ymm13
    vaddps	%ymm8, %ymm7, %ymm7
    vaddps	%ymm11, %ymm1, %ymm1
    vaddps	512(%rsp), %ymm12, %ymm2
    vaddps	%ymm9, %ymm5, %ymm5
    vaddps	%ymm15, %ymm6, %ymm6
    vaddps	%ymm10, %ymm4, %ymm4
    vaddps	%ymm8, %ymm7, %ymm7
    vaddps	%ymm11, %ymm1, %ymm1
    vaddps	%ymm12, %ymm2, %ymm2
    vaddps	%ymm14, %ymm0, %ymm0
    vmovaps	%ymm5, 384(%rsp)
    vmovaps	%ymm6, 480(%rsp)
    vaddps	608(%rsp), %ymm3, %ymm3
    vmovaps	%ymm4, 352(%rsp)
    vaddps	%ymm8, %ymm7, %ymm6
    vaddps	%ymm12, %ymm2, %ymm2
    vaddps	%ymm14, %ymm0, %ymm0
    vaddps	%ymm13, %ymm3, %ymm3
    vmovaps	%ymm6, 448(%rsp)
    vaddps	%ymm5, %ymm9, %ymm6
    vaddps	%ymm4, %ymm10, %ymm5
    vmovaps	%ymm0, 320(%rsp)
    vaddps	%ymm11, %ymm1, %ymm4
    vaddps	%ymm0, %ymm14, %ymm0
    vaddps	%ymm13, %ymm3, %ymm13
    vmovaps	%ymm4, 576(%rsp)
    vaddps	%ymm12, %ymm2, %ymm4
    vmovaps	%ymm13, 544(%rsp)
    vaddps	480(%rsp), %ymm15, %ymm13
```