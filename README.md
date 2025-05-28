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

