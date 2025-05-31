#include <iostream>
#include <chrono>
#include <cstdint>
#include <x86intrin.h>

int main() {

    int num_runs = 5;
    const uint64_t M = 100000, N = 1000000;
    alignas(32) float a[8] = {};
    for (int i = 0; i < 8; i++) a[i] = rand() / float(RAND_MAX) * 2.0f - 1.0f;
    float *out = (float *)malloc(M * sizeof(float));

    float b = 0.5f;

    double seconds = 0;
    unsigned long long cycles = 0;

    for(int run=0; run<num_runs; run++) {
        unsigned int aux;
        unsigned long long cyc0 = __rdtscp(&aux);
        auto t0 = std::chrono::high_resolution_clock::now();
    
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < M; i++) {
            alignas(32) float d[8] = {};
            #pragma unroll(8)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < 8; k++)
                    d[k] = a[k] * b + d[k];
            float s = 0;
            for (int k = 0; k < 8; k++) s += d[k];
            out[i] += s;
        }
    
        auto t1 = std::chrono::high_resolution_clock::now();
        unsigned long long cyc1 = __rdtscp(&aux);
        std::chrono::duration<double> wall = t1 - t0;
        seconds += wall.count();
        cycles += cyc1 - cyc0;
    }
    
    seconds /= num_runs;
    cycles /= num_runs;

    double total_flops = double(M) * double(N) * 16.0 * 1e-9; // GFLOPs
    double gflops = total_flops / seconds;
    double freq_hz = cycles / seconds;
    double freq_ghz = freq_hz / 1e9;

    std::cout<<"Wall time            : "<<seconds<<" s\n";
    std::cout<<"Total FLOPs          : "<<total_flops<<" GFLOP\n";
    std::cout<<"Achieved FLOPS       : "<<gflops<<" GFLOP/s\n";
    std::cout<<"Cycles elapsed       : "<<cycles<<"\n";
    std::cout<<"Measured CPU freq    : "<<freq_ghz<<" GHz\n";
    std::cout<<"Checksum             : "<<out[0]<<"\n";

    free(out);
    return 0;
}
