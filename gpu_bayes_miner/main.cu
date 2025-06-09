#include <iostream>
#include <cuda_runtime.h>

#define N 2  // Change as needed, e.g., 16, 32

// CUDA Kernel
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main() {
    float A[N][N], B[N][N], C[N][N];
    float (*d_A)[N], (*d_B)[N], (*d_C)[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i;
            B[i][j] = j;
        }

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel call
    dim3 threadsPerBlock(N, N);
    MatAdd<<<1, threadsPerBlock>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << C[i][j] << " ";
        std::cout << "\n";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
