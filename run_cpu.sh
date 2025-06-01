export OMP_NUM_THREADS=24
export OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}"
export OMP_PROC_BIND=close

g++ -fopenmp -g -O3 -march=native cp3b.cc -o main

perf stat ./main