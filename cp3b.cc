#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <chrono>
#include <cstdint>

constexpr int NUM_LOADS = 8;
// micro kernel
constexpr int MR = 8;
constexpr int NR = 16;
// for cache blocks
// MC and NC have to be multiple of NUM_LOADS
constexpr int MC = 256; // Load ndata[ic: ic + MC][:] in L3 cache block
constexpr int KC = 224;
constexpr int NC = 128;
// Padding Py has to be multiple of 16 as well as a multiple of MC
constexpr int Px = 1;
constexpr int Py = 16;

struct Matrix {
    int ny, nx;
    int nyd, nxc;
    const float *data;
    float *ndata;
    float *output;

    Matrix(int ny, int nx, const float *data) 
    : ny(ny), nx(nx), data(data) {
        nxc = ((nx + Px - 1) / Px) * Px;
        nyd = ((ny + Py - 1) / Py) * Py;
        ndata = (float *)aligned_alloc(32, nyd * nxc * sizeof(float));
        output = (float *)aligned_alloc(32, nyd * nyd * sizeof(float));

        #pragma omp parallel for
        for(int i=0; i<nyd*nyd; i++) output[i] = 0.0f;
    }
    
    inline void standardize_rows_swizzle() {

        #pragma omp parallel for
        for(int y=0; y<nyd; y+=NUM_LOADS) {

            double mean[NUM_LOADS] = {};
            float stdn[NUM_LOADS] = {};
            double d, delta;

            // 1) Compute mean and (unnormalized) variance per row in this block
            for(int j=0; j<NUM_LOADS && j + y < ny; j++) {
                for(int x=0; x<nx; x++) {
                    d = (double)data[x + nx * (y + j)];
                    delta = d - mean[j];
                    mean[j] += delta / (x + 1);
                    stdn[j] += delta * (d - mean[j]);
                }
                stdn[j] = 1.0f / sqrt(stdn[j]);
            }
    
            // 2) Normalize and write into swizzled layout
            int x=0;
            for(; x<nx; x++) {
                for(int j=0; j<NUM_LOADS; j++) {
                    if(y + j < ny) {
                        ndata[(y/NUM_LOADS) * NUM_LOADS * nxc + x * NUM_LOADS + j] = (data[x + nx * (y + j)] - mean[j]) * stdn[j];
                    } else {
                        ndata[(y/NUM_LOADS) * NUM_LOADS * nxc + x * NUM_LOADS + j] = 0.0f;
                    }
                }
            }
            // 3) Zero-pad any extra columns in swizzled block
            for(; x<nxc; x++) {
                for(int j=0; j<NUM_LOADS; j++) {
                    ndata[(y/NUM_LOADS) * NUM_LOADS * nxc + x * NUM_LOADS + j] = 0.0f;
                }
            }


        }

    }

    inline void covariance_tiled() const {
    
        #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for(int ic=0; ic<nyd; ic+= MC) for(int jc=0; jc<nyd; jc+=NC) {
            if(ic > jc) continue;
    
            for (int kc = 0; kc < nxc; kc += KC) {
                int Kb = std::min(KC, nxc - kc);
                int Mb = std::min(MC, nyd - ic);
                int Nb = std::min(NC, nyd - jc);
    
                for (int ib = 0; ib < Mb; ib += MR) {
                    for (int jb = 0; jb < Nb; jb += NR) {
                        if(ic == jc && ib - NR > jb) continue;

                        __m256 psum[NR/8][MR] = {};
    
                        float *blocki = ndata + (ic + ib)/NUM_LOADS * NUM_LOADS * nxc + kc * NUM_LOADS;
                        float *blockj = ndata + (jc + jb)/NUM_LOADS * NUM_LOADS * nxc + kc * NUM_LOADS;
    
                        for (int k = 0; k < Kb; k++) {

                            __m256 b0 = _mm256_load_ps(blockj + k * NUM_LOADS);
                            __m256 b1 = _mm256_load_ps(blockj + k * NUM_LOADS + NUM_LOADS * nxc);
                            for(int ik=0; ik<MR; ik++) {
                                __m256 a = _mm256_set1_ps(*(blocki + k * NUM_LOADS + ik));
                                psum[0][ik] = _mm256_fmadd_ps(
                                    b0,
                                    a,
                                    psum[0][ik]
                                );
                                psum[1][ik] = _mm256_fmadd_ps(
                                    b1,
                                    a,
                                    psum[1][ik]
                                );
                            }
    
                        }

                        for(int ik=0; ik<MR; ik++) {
                            float *loc_ptr = output + (ic + ib + ik) * nyd + jc + jb;
                            _mm256_store_ps(
                                loc_ptr,
                                _mm256_add_ps(_mm256_load_ps(loc_ptr), psum[0][ik])
                            );
                            _mm256_store_ps(
                                loc_ptr + NUM_LOADS,
                                _mm256_add_ps(_mm256_load_ps(loc_ptr + NUM_LOADS), psum[1][ik])
                            );
                        }
    
                    }
                }
    
            }
    
        }

    }

    inline void storeResult(float *result) {
        #pragma omp parallel for collapse(2)
        for(int i=0; i<ny; i++) {
            for(int j=0; j<ny; j++) {
                if (i > j) continue;
                result[i * ny + j] = output[i * nyd + j];
            }
        }
    }

    ~Matrix() {
        free(ndata);
        free(output);
    }

};

void correlate(int ny, int nx, const float *data, float *result) {

    Matrix m{ny, nx, data};

    m.standardize_rows_swizzle();

    int num_runs = 1;
    double seconds = 0;
    unsigned long long cycles = 0;

    for(int run=0; run<num_runs; run++) {
        unsigned int aux;
        unsigned long long cyc0 = __rdtscp(&aux);
        auto t0 = std::chrono::high_resolution_clock::now();
    
        m.covariance_tiled();
    
        auto t1 = std::chrono::high_resolution_clock::now();
        unsigned long long cyc1 = __rdtscp(&aux);
        std::chrono::duration<double> wall = t1 - t0;
        seconds += wall.count();
        cycles += cyc1 - cyc0;
    }

    seconds /= num_runs;
    cycles /= num_runs;

    double total_flops = double(ny) * double(ny) * double(nx) * 1e-9; // GFLOPs
    double gflops = total_flops / seconds;
    double freq_hz = cycles / seconds;
    double freq_ghz = freq_hz / 1e9;

    std::cout<<"Wall time            : "<<seconds<<" s\n";
    std::cout<<"Total FLOPs          : "<<total_flops<<" GFLOP\n";
    std::cout<<"Achieved FLOPS       : "<<gflops<<" GFLOP/s\n";
    std::cout<<"Cycles elapsed       : "<<cycles<<"\n";
    std::cout<<"Measured CPU freq    : "<<freq_ghz<<" GHz\n";
    std::cout<<"Checksum             : "<<m.output[10]<<"\n";

    m.storeResult(result);

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

    free(result);
    free(data);
    return 0;

}