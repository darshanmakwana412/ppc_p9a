#include <cstdio>
#include <cstdlib>
#include <cuda.h>

extern "C" __global__
void gpu_kernel(
    float *out,
    int iters, float seed
) {

    float a = seed + threadIdx.x;
    float b = seed - threadIdx.x;

    #pragma unroll (100)
    for (int i = 0; i < iters; i++) {
        a = a * b + a;
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main() {

    int iters = 1e6;
    float seed = 0.869f;

    int threads = 256;
    int blocks = 1024;
    size_t N = blocks * threads;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_kernel<<<blocks,threads>>>(
        d_out, iters, seed
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);

    double total_flops = double(iters) * blocks * threads * 2 / 1e9;
    double secs = msec / 1000.0;
    double gflops = total_flops / secs;

    printf("Elapsed Time : %.3f ms (%.3f s)\n", msec, secs);
    printf(" Total FLOP  : %.3e GFLOP\n", total_flops);
    printf("Performance  : %.3f GFLOP/s\n", gflops);

    cudaFree(d_out);
    return 0;

}
