#include <iostream>
#include <chrono>
#include <cstdint>
#include <x86intrin.h>
#include <cstdlib>
#include <ctime>
#include <omp.h>

int main() {

    int num_runs = 5;
    const uint64_t M = 100000, N = 500000;
    alignas(32) float a[8] = {};
    for (int i = 0; i < 8; i++) a[i] = rand() / float(RAND_MAX) * 2.0f - 1.0f;
    float *out = (float *)malloc(M * sizeof(float));

    float b = 0.5f;
    __m256 a8 = _mm256_load_ps(a);
    __m256 b8 = _mm256_set1_ps(b);

    double seconds = 0;
    unsigned long long cycles = 0;

    for(int run=0; run<num_runs; run++) {
        unsigned int aux;
        unsigned long long cyc0 = __rdtscp(&aux);
        auto t0 = std::chrono::high_resolution_clock::now();
    
        #pragma omp parallel for schedule(dynamic, 1)
        for (uint64_t i = 0; i < M; i++) {

            __m256 d8[8] = {};
            for(int k=0; k<8; k++) {
                // randomly initialize d8[k]
                alignas(32) float v[8] = {};
                for (int j = 0; j < 8; j++) v[j] = rand() / float(RAND_MAX) * 2.0f - 1.0f;
                d8[k] = _mm256_load_ps(v);
            }
            for (uint64_t j = 0; j < N; j++) {
                #pragma GCC unroll 8
                for(int k=0; k<8; k++) {
                    d8[k] = _mm256_fmadd_ps(a8, b8, d8[k]);
                }
            }

            // Since N is much larger the flops of this is negligible
            __m256 s = _mm256_setzero_ps();
            for(int i=0; i<8; i++) {
                s = _mm256_add_ps(s, d8[i]);
            }
            out[i] += s[0];
        }
    
        auto t1 = std::chrono::high_resolution_clock::now();
        unsigned long long cyc1 = __rdtscp(&aux);
        std::chrono::duration<double> wall = t1 - t0;
        seconds += wall.count();
        cycles += cyc1 - cyc0;
    }
    
    seconds /= num_runs;
    cycles /= num_runs;

    double total_flops = double(M) * double(N) * 128.0 * 1e-9; // GFLOPs
    double gflops = total_flops / seconds;
    double freq_hz = cycles / seconds;
    double freq_ghz = freq_hz / 1e9;
    double sum = 0;
    for (int i = 0; i < 8; i++) sum += a[i];

    std::cout<<"Avg Wall time        : "<<seconds<<" s\n";
    std::cout<<"Total FLOPs          : "<<total_flops<<" GFLOP\n";
    std::cout<<"Avg Achieved FLOP    : "<<gflops<<" GFLOP/s\n";
    std::cout<<"Cycles elapsed       : "<<cycles<<"\n";
    std::cout<<"Measured CPU freq    : "<<freq_ghz<<" GHz\n";
    std::cout<<"sum                  : "<<sum<<"\n";

    free(out);
    return 0;
}
