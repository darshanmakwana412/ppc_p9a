export OMP_NUM_THREADS=24

g++ -fopenmp -g -O3 -march=native -ffast-math -funroll-loops cpu_flops.cpp -o main
# g++ -fopenmp -O3 -march=native -S cpu_flops.cpp -o main.s

perf stat ./main