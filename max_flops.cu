#include <cstdio>
#include <cstdlib>
#include <cuda.h>

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

    if(tid == 0) out[blockIdx.x * blockDim.x + tid] = a;
}

int main() {

    int N = 1e7;
    float seed = 0.869f;

    int threads = 256;
    int blocks = 1024;
    size_t size = blocks * threads;
    float *d_out;
    cudaMalloc(&d_out, size * sizeof(float));

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_kernel<<<blocks,threads>>>(
        d_out, N, seed
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);

    double total_flops = double(N) * blocks * threads * 2 / 1e9;
    double secs = msec / 1000.0;
    double gflops = total_flops / secs;

    printf("Elapsed Time : %.3f ms (%.3f s)\n", msec, secs);
    printf(" Total FLOP  : %.3e GFLOP\n", total_flops);
    printf("Performance  : %.3f GFLOP/s\n", gflops);

    cudaFree(d_out);
    return 0;

}
