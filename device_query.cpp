#include <cuda_runtime.h>
#include <iostream>

// https://siboehm.com/articles/22/CUDA-MMM
void printOccupancyStats() {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "===== Device " << dev << " =====\n";
        std::cout << "Name: "                << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Max threads per block: "         << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Threads per warp: "              << prop.warpSize << "\n";
        std::cout << "Max registers per block: "       << prop.regsPerBlock << "\n";
        std::cout << "Max registers per multiprocessor: " << prop.regsPerMultiprocessor << "\n";
        std::cout << "Total global memory: "          << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "Max shared mem per block: "     << (prop.sharedMemPerBlock >> 10) << " KB\n";
        std::cout << "Shared mem per multiprocessor: " << (prop.sharedMemPerMultiprocessor) << " B\n";
        std::cout << "Multiprocessor count: "         << prop.multiProcessorCount << "\n";
        std::cout << "Max warps per multiprocessor: " 
                  << (prop.maxThreadsPerMultiProcessor / prop.warpSize) << "\n\n";
    }
}

int main() {

    printOccupancyStats();
    return 0;

}