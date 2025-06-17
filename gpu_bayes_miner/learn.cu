#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#include <stdio.h>

__global__ void increment_gpu(long long int *array, long long int N) {
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
    printf("here is %lld\n", N);
    if (i < N) {
        array[i]++;
    }
}

int main() {
    long long int N = 1000000000;
    long long int* host_array = new long long int[N];

    for (long long int i = 0; i < N; ++i)
        host_array[i] = i;

    long long int* device_array;
    cudaMalloc((void**)&device_array, N * sizeof(long long int));
    cudaMemcpy(device_array, host_array, N * sizeof(long long int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    increment_gpu<<<blocksPerGrid, threadsPerBlock>>>(device_array, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host_array, device_array, N * sizeof(long long int), cudaMemcpyDeviceToHost);

    cudaFree(device_array);
    delete[] host_array;

    return 0;
}
