#include <iostream>
#include <chrono>
#include <cstdint>
#include <x86intrin.h>

int main() {

    const uint64_t M = 100000, N = 100000;
    float a[8] = {};
    for (int i = 0; i < 8; i++) a[i] = rand() / float(RAND_MAX) * 2.0f - 1.0f;

    float b = 0.5f;

    unsigned int aux;
    unsigned long long cyc0 = __rdtscp(&aux);
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < M; i++) {
        float d[8] = {};
        for (int j = 0; j < N; j++)
            for (int k = 0; k < 8; k++)
                d[k] = a[k] * b + d[k];
        for (int k = 0; k < 8; k++) a[k] += d[k];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    unsigned long long cyc1 = __rdtscp(&aux);

    std::chrono::duration<double> wall = t1 - t0;
    double seconds = wall.count();
    unsigned long long cycles = cyc1 - cyc0;

    double total_flops = double(M) * double(N) * 16.0 * 1e-9; // GFLOPs
    double gflops = total_flops / seconds;
    double freq_hz = cycles / seconds;
    double freq_ghz = freq_hz / 1e9;

    std::cout<<"Wall time            : "<<seconds<<" s\n";
    std::cout<<"Total FLOPs          : "<<total_flops<<" GFLOP\n";
    std::cout<<"=> Achieved          : "<<gflops<<" GFLOP/s\n";
    std::cout<<"Cycles elapsed       : "<<cycles<<"\n";
    std::cout<<"=> Measured CPU freq : "<<freq_ghz<<" GHz\n";
    std::cout<<"Checksum             : "<<(a[0]+a[1]+a[2]+a[3]+a[4]+a[5]+a[6]+a[7])<<"\n";

    return 0;
}
