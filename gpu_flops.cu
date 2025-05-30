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
    float c = seed * a;
    float d = seed * b;

    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int u = 0; u < 16; u++) {
            a = fmaf(a, c, a);
            b = fmaf(b, d, b);
        }
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a + b + c + d;
}

int main() {

    int iters = 1e7;
    float seed = 0.869f;

    int threads = 256;
    int blocks = 80;
    size_t N = blocks * threads;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // warm up
    gpu_kernel<<<blocks,threads>>>(
        d_out, 10, seed
    );
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

    double total_flops = double(iters) * 64.0 * blocks * threads / 1e9;
    double secs = msec / 1000.0;
    double gflops = total_flops / secs;

    printf("Elapsed Time : %.3f ms (%.3f s)\n", msec, secs);
    printf(" Total FLOP  : %.3e GFLOP\n", total_flops);
    printf("Perforamance : %.3f GFLOP/s\n", gflops);

    cudaFree(d_out);
    return 0;

}
