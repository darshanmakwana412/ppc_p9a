## Theoretical limitations

For the GPU running `nvcc device_query.cpp -o main && ./main` gives
```bash
Name: NVIDIA GeForce RTX 2050
Compute Capability: 8.6
Clock Rate: 1.155 GHz 
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
Thus the GPU has 16 streaming multiprocessors (SM) and from the wiki of [GeForce RTX 2050](https://en.wikipedia.org/wiki/GeForce_RTX_20_series#Laptop) it has 2048 CUDA cores and 64 Tensor Cores. It also has a bandwidth of 112 GB/s and a default clock freq of 1.155 GHz and boost of 1.477 GHz. Each core can execute 1 fma i.e 2 FLOP per cycle, we can then calculate the theoretical max flops as follows

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

[1] - [Nvidia GeForce RTX datasheet](https://www.techpowerup.com/gpu-specs/geforce-rtx-2050-mobile.c3859)

## Practical limitations

For doing max computation on gpu we will let each thread execute $N$ fma i.e each thread will compute $a = a *b + a$. Each thread then does a total of $2 * N$ flops which will allow the entire kernel to do blocks * threads * 2 * N flops. Here is our very simple and tiny kernel, the entire code is available at [max_flops.cu](max_flops.cu)
```cpp
extern "C" __global__
void gpu_kernel(
    float *out,
    int N, float seed
) {

    int tid = threadIdx.x;
    float a = seed + tid;
    float b = seed - tid;

    for (int i = 0; i < N; i++) {
        a = a * b + a;
    }

    out[blockIdx.x * blockDim.x + tid] = a;
}

gpu_kernel<<<1024, 256>>>(
    d_out, iters, seed
);
```

We run this with `nvcc max_flops.cu -O3 -arch=sm_86 -o main && ./main` with N set to `1e7` and it gives the following output
```bash
Elapsed Time : 1132.182 ms (1.132 s)
 Total FLOP  : 5.243e+03 GFLOP
Performance  : 4630.773 GFLOP/s
```

We thus achieve 97.88% of the theoretical maximum flops. Profiling it with `sudo ncu ./main` gives us
```bash
gpu_kernel (1024, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ----------------
    Metric Name               Metric Unit     Metric Value
    ----------------------- ------------- ----------------
    DRAM Frequency          cycle/nsecond             6.00
    SM Frequency            cycle/usecond           832.57
    Elapsed Cycles                  cycle    1,532,524,284
    Memory Throughput                   %             0.06
    DRAM Throughput                     %             0.00
    Duration                       second             1.13
    L1/TEX Cache Throughput             %             0.00
    L2 Cache Throughput                 %             0.06
    SM Active Cycles                cycle 1,532,183,860.38
    Compute (SM) Throughput             %            99.18
    ----------------------- ------------- ----------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing workloads in the Compute Workload Analysis section.                                        

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
    Achieved Occupancy                        %        95.84
    Achieved Active Warps Per SM           warp        46.00
    ------------------------------- ----------- ------------

    INF   This kernel's theoretical occupancy is not impacted by any block limit.          
```
We can see that our kernel achieved a compute capacity of `99.18%` which is all thanks to ~0% memory utilization. We also achieved 95.84% of max occupancy with 46 active warps out of 48

Looking at the ptx code with `nvcc --ptx -arch=sm_86 max_flops.cu -o main.ptx` we see that the compiler directly generated fma f32 instructions while unrolling the inner loop by 4 for us
```assembly
$L__BB0_3:
	fma.rn.f32 	%f13, %f2, %f20, %f20;
	fma.rn.f32 	%f14, %f2, %f13, %f13;
	fma.rn.f32 	%f15, %f2, %f14, %f14;
	fma.rn.f32 	%f20, %f2, %f15, %f15;
	add.s32 	%r13, %r13, -4;
	setp.ne.s32 	%p3, %r13, 0;
	@%p3 bra 	$L__BB0_3;
```
The executable can be found at [max_flops](max_flops) and the ptx code at [max_flops.ptx](max_flops.ptx)

## Comparison with CP5 Solution

Now for comparison with the fastest [cp5.cu](cp5.cu) solution we again benchmark using `nvcc cp5.cu -O3 -arch=sm_86 -o main && ./main`. It follows the exact same code as submitted to cp5 without any modification and nx and ny set to `14000`, the executable can be found at [cp5](cp5)
```cpp
Elapsed Time : 1050.230 ms (1.050 s)
 Total FLOP  : 2.744e+03 GFLOP
Performance  : 2612.948 GFLOP/s
```
It executated $2\times 14\times 14\times 14  / 2= 2744$ GFLOP (as we are only computing the upper triangular matrix) in 1.050s achieving 2612.948 GFLOP/s which is 55.23% of the theoretical peak flops @ base. Profiling it with `sudo ncu ./main` reveals

```bash
matmul_kernel_v3 (110, 110, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ----------------
    Metric Name               Metric Unit     Metric Value
    ----------------------- ------------- ----------------
    DRAM Frequency          cycle/nsecond             6.00
    SM Frequency            cycle/usecond           832.53
    Elapsed Cycles                  cycle    1,069,374,085
    Memory Throughput                   %            56.61
    DRAM Throughput                     %            54.71
    Duration                       second             1.28
    L1/TEX Cache Throughput             %            56.73
    L2 Cache Throughput                 %            33.48
    SM Active Cycles                cycle 1,067,132,546.81
    Compute (SM) Throughput             %            69.46
    ----------------------- ------------- ----------------

    WRN   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the   
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and      
          could be reduced or moved to look-up tables.                                                                  

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 12,100
    Registers Per Thread             register/thread             100
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            8.19
    Threads                                   thread       3,097,600
    Waves Per SM                                              378.12
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            2
    Block Limit Shared Mem                block            7
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %        33.33
    Achieved Occupancy                        %        33.30
    Achieved Active Warps Per SM           warp        15.98
    ------------------------------- ----------- ------------

    WRN   This kernel's theoretical occupancy (33.3%) is limited by the number of required registers. See the CUDA Best 
          Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more      
          details on optimizing occupancy.                                                                   
```

This time we achieved 70% of max compute throughput and only 33% max theoretical occupancy which is due to high register use of 100 per thread and 16 active warps per block possible and also our memory throughput is 55% of peak. Thus we are register bound

Looking at the ptx code from [cp5.ptx](cp5.ptx) we see that the compiler issues vectorized loads from shared memory this time
```assembly
	fma.rn.f32 	%f701, %f636, %f617, %f605;
	fma.rn.f32 	%f702, %f636, %f618, %f606;
	fma.rn.f32 	%f703, %f636, %f619, %f607;
	fma.rn.f32 	%f704, %f636, %f620, %f608;
	ld.shared.v4.f32 	{%f705, %f706, %f707, %f708}, [%r74+2048];
	ld.shared.v4.f32 	{%f713, %f714, %f715, %f716}, [%r74+2304];
	ld.shared.v4.f32 	{%f721, %f722, %f723, %f724}, [%r77+2048];
	ld.shared.v4.f32 	{%f729, %f730, %f731, %f732}, [%r77+2304];
	fma.rn.f32 	%f737, %f721, %f705, %f641;
	fma.rn.f32 	%f738, %f721, %f706, %f642;
	fma.rn.f32 	%f739, %f721, %f707, %f643;
```

not only that we can also see that the compiler also generated code for vectorized load from global to shared memory
```assembly
$L__BB1_3:
	.loc	1 108 9
	ld.global.nc.v4.u32 	{%r56, %r57, %r58, %r59}, [%rd39];
	st.shared.v4.u32 	[%r6], {%r56, %r57, %r58, %r59};
	.loc	1 109 9
	ld.global.nc.v4.u32 	{%r64, %r65, %r66, %r67}, [%rd38];
	st.shared.v4.u32 	[%r5], {%r64, %r65, %r66, %r67};
```