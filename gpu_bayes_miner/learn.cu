#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void increment_gpu(long long int *array, long long int N) {
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
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

    for (int i = 0; i < 10; ++i) {  // Print first 10 for readability
        cout << host_array[i] << " ";
    }
    cout << endl;

    cudaFree(device_array);
    delete[] host_array;

    return 0;
}
