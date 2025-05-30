# P9a

## Theoretical limitations

For the cpu running `lscpu` gives
```bash
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   16
  On-line CPU(s) list:    0-15
Vendor ID:                GenuineIntel
  Model name:             12th Gen Intel(R) Core(TM) i5-1240P
    CPU family:           6
    Model:                154
    Thread(s) per core:   2
    Core(s) per socket:   12
    Socket(s):            1
    Stepping:             3
    CPU(s) scaling MHz:   21%
    CPU max MHz:          4400.0000
    CPU min MHz:          400.0000
    BogoMIPS:             4224.00
    Flags:                avx avx2 avx_vnni fma ...
Caches (sum of all):      
  L1d:                    448 KiB (12 instances)
  L1i:                    640 KiB (12 instances)
  L2:                     9 MiB (6 instances)
  L3:                     12 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-15
```
It shows that the cpu is 12th Gen Intel(R) Core(TM) i5-1240P and has 16 cores (8 Efficient cores and 4 Performant cores with 2 hyper threading threads) and operates at a clock frequency of [1.7GHz by default](https://www.techpowerup.com/cpu-specs/core-i5-1240p.c2583) and can be max operated at 4.4GHz, it also supports upto avx2 thus it has a per core per cycle FLOPS of 4 (4 doubles per 256 bit lane) * 2 (fma) = 8 for double precision 8 * 2 = 16 for single precision 

$$
\begin{aligned}
\text{DP peak @ default} 
&= 16 \times 8 \times 1.7\!\times\!10^9 
\approx 217.6\;\mathrm{GFLOP/s}\\
\text{DP peak @ max} 
&= 16 \times 8 \times 4.4\!\times\!10^9 
\approx 563.2\;\mathrm{GFLOP/s}
\end{aligned}
$$

$$
\begin{aligned}
\text{SP peak @ default} 
&= 16 \times 16 \times 1.7\!\times\!10^9 
\approx 435.2\;\mathrm{GFLOP/s}\\
\text{SP peak @ max} 
&= 16 \times 16 \times 4.4\!\times\!10^9 
\approx 1126.4\;\mathrm{GFLOP/s}
\end{aligned}
$$

For the GPU running running `nvcc device_query.cpp -o main && ./main` gives
```bash
===== Device 0 =====
Name: NVIDIA GeForce RTX 2050
Compute Capability: 8.6
Max threads per block: 1024
Max threads per multiprocessor: 1536
Threads per warp: 32
Max registers per block: 65536
Max registers per multiprocessor: 65536
Total global memory: 3897 MB
Max shared mem per block: 48 KB
Shared mem per multiprocessor: 102400 B
Multiprocessor count: 16
Max warps per multiprocessor: 48
```
Thus the GPU has 16 streaming multiprocessors (SM) and from the wiki of [GeForce RTX 2050](https://en.wikipedia.org/wiki/GeForce_RTX_20_series#Laptop) it has 2048 CUDA cores and 64 Tensor Cores. It also has a bandwidth of 112 GB/s and a default clock freq of 1.155 GHz and boost of 1.477 GHz. Each core can execute 1 fma i.e 2 FLOP per cycle

$$
\begin{aligned}
\text{SP peak @ default} 
&= 2048 \times 2 \times 1.155\!\times\!10^9 
\approx 4730.88\;\mathrm{GFLOP/s}\\
\text{SP peak @ boost} 
&= 2048 \times 2 \times 1.477\!\times\!10^9 
\approx 6049.79\;\mathrm{GFLOP/s}
\end{aligned}
$$

This matches with the processing power with boost in the wiki of GeForce 2050

## Practical limitations of CPU

For this we will initialize a vector of size `a` of size 8 (so the compiler can easily auto vectorize the inner for loop) and then mutiply it with a scalar `b` and add it back to itself. We will do this $M * N$ times. The main part of the code is very straight forward and simple we do $a[k] = a[k] \times b + a[k], \forall k\in [8]$. This code does a total of $16MN + 16N$ flops, since M is of the order of `1e5` we will ignore the `16N`. This gives us a total of `16MN` flops
```cpp
#pragma omp parallel for
for (int i = 0; i < M; i++) {
    float d[8] = {};
    for (int j = 0; j < N; j++)
        for (int k = 0; k < 8; k++)
            d[k] = a[k] * b + d[k];

    for (int k = 0; k < 8; k++) a[k] += d[k];
}
```

We will then time how long it takes to run this using `std::chrono::high_resolution_clock` and we will also measure the clock cycles during it's execution using `__rdtscp` intrinsics this will help us get the measured clock freq. The entire code is at [cpu_flops.cpp](./cpu_flops.cpp) we will run this with aggressive auto vectorization and loop unrolling. [run_cpu.sh](./run_cpu.sh) is a small bash script which runs this code here are the flags that are enabled in it. We also use perf stat for analyzing the instructions and branches and misses

```bash
export OMP_NUM_THREADS=16
g++ -fopenmp -O3 -march=native -ffast-math -funroll-loops max_flops.cpp -o main
perf stat ./main
```

On running `source run_flops.sh` it prints this
```bash
Wall time            : 3.12342 s
Total FLOPs          : 1600 GFLOP
=> Achieved          : 512.26 GFLOP/s
Cycles elapsed       : 6596566797
=> Measured CPU freq : 2.11197 GHz
Checksum             : inf
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

## Practical limitations of GPU

