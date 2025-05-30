## Theoretical limitations

For the GPU running `nvcc device_query.cpp -o main && ./main` gives
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
Thus the GPU has 16 streaming multiprocessors (SM) and from the wiki of [GeForce RTX 2050](https://en.wikipedia.org/wiki/GeForce_RTX_20_series#Laptop) it has 2048 CUDA cores and 64 Tensor Cores. It also has a bandwidth of 112 GB/s and a default clock freq of 1.155 GHz and boost of 1.477 GHz. Each core can execute 1 fma i.e 2 FLOP per cycle, we can the following max single precision flops at base and boost

$$
\begin{aligned}
\text{SP peak @ base} 
&= 2048 \times 2 \times 1.155\!\times\!10^9 
\approx 4730.88\;\mathrm{GFLOP/s}\\
\text{SP peak @ boost} 
&= 2048 \times 2 \times 1.477\!\times\!10^9 
\approx 6049.79\;\mathrm{GFLOP/s}
\end{aligned}
$$

This matches with the processing power @ boost in the wiki of GeForce 2050

## Practical limitations of GPU

For doing max computation on gpu each thread will do `iters` number of `fmaf` with a and b i.e we do a = a * b + a iters times. Each thread does a total of iters * 2 flops, we do a total of blocks * threads * iters * 2 flops. Here is our kernel
```cpp
extern "C" __global__
void gpu_kernel(
    float *out,
    int iters, float seed
) {

    float a = seed + threadIdx.x;
    float b = seed - threadIdx.x;

    for (int i = 0; i < iters; i++) {
        a = fmaf(a, b, a);
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

gpu_kernel<<<blocks,threads>>>(
    d_out, iters, seed
);
```

We run this with `nvcc -O3 -arch=sm_86 gpu_flops.cu -o main` using the script `source run_gpu.sh` gives the following output
```bash
Elapsed Time : 115.149 ms (0.115 s)
 Total FLOP  : 5.243e+02 GFLOP
Performance  : 4553.145 GFLOP/s
```

Porfilling it with `sudo ncu ./main` gives the following
```bash
gpu_kernel (1024, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         5.86
    SM Frequency            cycle/usecond       816.36
    Elapsed Cycles                  cycle       14,059
    Memory Throughput                   %        32.55
    DRAM Throughput                     %        32.55
    Duration                      usecond        17.22
    L1/TEX Cache Throughput             %        29.17
    L2 Cache Throughput                 %        30.76
    SM Active Cycles                cycle    12,480.75
    Compute (SM) Throughput             %        66.28
    ----------------------- ------------- ------------
                                                          
    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  1,024
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread         262,144
    Waves Per SM                                               10.67
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        79.73
    Achieved Active Warps Per SM           warp        38.27
    ------------------------------- ----------- ------------
```
The SM freqs shows to be 0.8613 GHz as compared to the base 1.155GHz on wiki which I have no idea why is this difference
We thus achieve 96.25% of the theoretical maximum flops

## Comparison with CP Solution

