nvcc -O3 -arch=sm_86 \
     -Xptxas -v \
     -keep \
     gpu_flops.cu -o main

./main