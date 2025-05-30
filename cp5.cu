#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

constexpr int NUM_COLS = 8;
constexpr int Px = NUM_COLS;
constexpr int Py = 128;

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


extern "C" __global__
void preprocess_kernel(
    const float* __restrict__ data,
    float* __restrict__ B,
    int ny, int nx,
    int nyp, int nxp
) {

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if(row >= ny) {
        for (int col = tid; col < nxp; col += stride) {
            B[row + col * nyp] = 0.0f;
        }
        return;
    }

    extern __shared__ double sh[];
    double *s_sum = sh;
    double *s_sumsq = sh + blockDim.x + 1;

    double sum = 0.0;
    double sumsq = 0.0;
    for (int col = tid; col < nx; col += stride) {
        double v = data[row * nx + col];
        sum += v;
        sumsq += v * v;
    }

    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();

    // parallel reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sumsq[tid] += s_sumsq[tid + offset];
        }
        __syncthreads();
    }

    double mean = s_sum[0] / nx;
    double inv_std = 1.0 / sqrt(s_sumsq[0] - mean * mean * nx);

    // Standardize, pad
    for (int col = tid; col < nxp; col += stride) {
        double out = 0.0;
        if (row < ny && col < nx) {
            double v = data[row * nx + col];
            out = (v - mean) * inv_std;
        }
        B[col * nyp + row] = out;
    }

}

extern "C" __global__
void matmul_kernel_v3(
    const float* __restrict__ B,
    float* __restrict__ C,
    int nyp, int nxp,
    int ny, int nx
) {

    int ic = blockIdx.x;
    int jc = blockIdx.y;

    if(ic > jc) return;

    int ia = threadIdx.x;
    int ja = threadIdx.y;

    float v[8][8] = {};
    __shared__ float xx[NUM_COLS * 128];
    __shared__ float yy[NUM_COLS * 128];

    int shift = (ja % 2) * 16 + ia;
    float4 *B4 = (float4 *)B + nyp * (ja / 2) / 4 + shift;
    int si = ic * 32;
    int sj = jc * 32;
    float4 *xx4 = (float4 *)xx + (ja / 2) * 32 + shift;
    float4 *yy4 = (float4 *)yy + (ja / 2) * 32 + shift;

    for (int ks = 0; ks < nxp; ks+=NUM_COLS) {

        xx4[0] = B4[si];
        yy4[0] = B4[sj];

        B4 += (NUM_COLS * nyp) / 4;

        __syncthreads();

        #pragma unroll
        for (int f = 0; f < NUM_COLS; ++f) {
            float y[8], x[8];

            ((float4 *)y)[0] = ((float4 *)yy)[f * 32 + ja];
            ((float4 *)y)[1] = ((float4 *)yy)[f * 32 + 16 + ja];
            ((float4 *)x)[0] = ((float4 *)xx)[f * 32 + ia];
            ((float4 *)x)[1] = ((float4 *)xx)[f * 32 + 16 + ia];

            for (int ib = 0; ib < 8; ++ib) {
                for (int jb = 0; jb < 8; ++jb) {
                    v[ib][jb] += x[ib] * y[jb];
                }
            }

        }

        __syncthreads();
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 128 + (ib / 4) * 64 + ia * 4 + (ib % 4);
            int j = jc * 128 + (jb / 4) * 64 + ja * 4 + (jb % 4);
            if (i < ny && j < ny && i <= j) {
                C[ny * i + j] = v[ib][jb];
            }
        }
    }

}

void correlate(int ny, int nx, const float *data, float *result) {

    int nyp = Py * ((ny + Py - 1) / Py);
    int nxp = Px * ((nx + Px - 1) / Px);

    float *d_data;
    cudaMalloc((void **)&d_data, ny * nx * sizeof(float));
    CHECK(cudaGetLastError());
    cudaMemcpy(d_data, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice);
    CHECK(cudaGetLastError());

    float *d_B, *d_C;
    cudaMalloc((void **)&d_B, nxp * nyp * sizeof(float));
    CHECK(cudaGetLastError());
    cudaMalloc((void **)&d_C, ny * ny * sizeof(float));
    CHECK(cudaGetLastError());

    cudaMemset(d_C, 0, ny * ny * sizeof(float));
    CHECK(cudaGetLastError());
    if(nxp > nx) {
        cudaMemset(d_B + nyp * nx, 0, nyp * (nxp - nx) * sizeof(float));
        CHECK(cudaGetLastError());
    }

    int threads = 1024;
    int shared = (1 + threads) * 2 * sizeof(double);
    dim3 grid(nyp);
    dim3 block(threads);
    preprocess_kernel<<<grid, block, shared>>>(
        d_data, d_B,
        ny, nx,
        nyp, nxp
    );
    CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dim3 grid1(nyp / 128, nyp / 128);
    dim3 block1(16, 16);
    matmul_kernel_v3<<<grid1, block1>>>(
        d_B, d_C,
        nyp, nxp, ny, nx
    );
    CHECK(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);

    double total_flops = (double)ny * (double)(ny + 1) * (double)nx / 1e9;
    double secs = msec / 1000.0;
    double gflops = total_flops / secs;

    printf("Elapsed Time : %.3f ms (%.3f s)\n", msec, secs);
    printf(" Total FLOP  : %.3e GFLOP\n", total_flops);
    printf("Performance  : %.3f GFLOP/s\n", gflops);

    cudaMemcpy(result, d_C, ny * ny * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    cudaFree(d_data);
    CHECK(cudaGetLastError());
    cudaFree(d_B);
    CHECK(cudaGetLastError());
    cudaFree(d_C);
    CHECK(cudaGetLastError());

}

int main() {

    int nx = 14000, ny = 14000;
    float *data = (float *)malloc(nx * ny * sizeof(float));
    float *result = (float *)malloc(ny * ny * sizeof(float));

    for(int i=0; i<ny; i++) {
        for(int j=0; j<nx; j++) {
            data[i * nx + j] = rand() / float(RAND_MAX) * 2.0f - 1.0f;
        }
    }

    correlate(
        ny, nx,
        data, result
    );

}