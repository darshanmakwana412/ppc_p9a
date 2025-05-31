export OMP_NUM_THREADS=24

g++ -fopenmp -g -O3 -march=native cpu_flops.cc -o main

perf stat ./main