#include <cstdio>
#include <cmath>
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <chrono>

#define BLOCKSIZE 32

//grid=> block=> thread
__global__ void testMul(int M, int N, int K, float * A, float * B, float * C, float alpha, float beta) {
	
	const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
	const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

	if (x < M && y < N) {
		float tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[x * K + i] * B[i * N + y];
		}
		C[x * N + y] = alpha * tmp + beta * C[x * N + y];
	}
}

__global__ void naive(int M, int N, int K, float * A, float * B, float * C, float alpha, float beta) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

__global__ void testTiledMatrix(int M, int N, int K, float * A, float * B, float * C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCKSIZE;
    const int BN = BLOCKSIZE;
    const int BK = BLOCKSIZE;
    
    // Convert 1D thread index to 2D coordinates within block
    int tx = threadIdx.x % BN;  // column in tile
    int ty = threadIdx.x / BN;  // row in tile

    // Shared memory for tiles
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // Move to current block position
    A = &A[by * BM * K];        // Start of block row in A
    B = &B[bx * BN];            // Start of block column in B  
    C = &C[by * BM * N + bx * BN]; // Start of block in C

    float tmp = 0.0f;
    
    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += BK) {
        // Load tile from A and B simultaneously
        // No boundary checks needed for square matrices divisible by BLOCKSIZE
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Move pointers for next iteration
        A += BK;
        B += BK * N;
        
        // Compute partial dot product using loaded tiles
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to global memory - no boundary check needed
    C[ty * N + tx] = tmp;
}

int main() {

    float *a, *b, *c, *c2;
    int M = 4096, N = 4096, K = 4096;

    cudaMallocManaged(&a, M*N*sizeof(float));
    cudaMallocManaged(&b, N*K*sizeof(float));
    cudaMallocManaged(&c, M*K*sizeof(float));
    cudaMallocManaged(&c2, M*K*sizeof(float));

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceSynchronize();

    std::cout << "Initializing matrices..." << std::endl;
    for(int i = 0 ; i < M; i++) 
        for (int j = 0; j < N ; ++j) 
            a[i*N + j] = rand() / (float)RAND_MAX;

    for(int i = 0 ; i < N; i++) 
        for (int j = 0; j < K ; ++j) 
            b[i*K + j] = rand() / (float)RAND_MAX;

    // Calculate theoretical FLOPS for comparison
    double flops = 2.0 * M * N * K;

    // Timing testMul kernel
    std::cout << "Running custom testMul kernel..." << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);


    float alpha = 1.0f;
    float beta = 0.0f;
    // For 1D thread blocks that map to 2D coordinates
    // Each block has BLOCKSIZE*BLOCKSIZE threads arranged as 1D
    dim3 threadsPerBlock(BLOCKSIZE * BLOCKSIZE, 1);
    dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
    std::cout << "Running custom tiled kernel..." << std::endl;
    testTiledMatrix<<<gridDim, threadsPerBlock>>>(M, N, K, a, b, c);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    float testMul_time;
    cudaEventElapsedTime(&testMul_time, start, stop);
    testMul_time /= 1000.0f; // Convert to seconds
    
    double testMul_gflops = flops / (testMul_time * 1e9);

    // Timing cuBLAS
    std::cout << "Running cuBLAS implementation..." << std::endl;
    float alpha_cublas = 1.0f;
    float beta_cublas = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEventRecord(start);
    // For row-major matrices A(MxN) * B(NxK) = C(MxK)
    // cuBLAS computes C = alpha * op(A) * op(B) + beta * C
    // Since cuBLAS expects column-major, we compute: C^T = B^T * A^T
    // This gives us C = A * B in row-major format
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                K, M, N, 
                &alpha_cublas, 
                b, K,  // B is NxK, leading dimension K in row-major 
                a, N,  // A is MxN, leading dimension N in row-major
                &beta_cublas, 
                c2, K  // C is MxK, leading dimension K in row-major
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_time;
    cudaEventElapsedTime(&cublas_time, start, stop);
    cublas_time /= 1000.0f; // Convert to seconds
    
    double cublas_gflops = flops / (cublas_time * 1e9);

    // Print performance comparison
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << " * " << N << "x" << K << std::endl;
    std::cout << "Total FLOPS: " << flops / 1e9 << " GFLOPS" << std::endl;
    std::cout << "\nCustom testMul kernel:" << std::endl;
    std::cout << "  Time: " << testMul_time * 1000 << " ms" << std::endl;
    std::cout << "  Performance: " << testMul_gflops << " GFLOPS" << std::endl;
    std::cout << "\ncuBLAS implementation:" << std::endl;
    std::cout << "  Time: " << cublas_time * 1000 << " ms" << std::endl;
    std::cout << "  Performance: " << cublas_gflops << " GFLOPS" << std::endl;
    std::cout << "\nSpeedup: " << (testMul_time / cublas_time) << "x faster with cuBLAS" << std::endl;

    // Verify correctness with a small sample
    std::cout << "\nVerifying correctness (checking first 10x10 submatrix)..." << std::endl;
    float eps = 1e-3; // Relaxed epsilon for large matrices
    int mismatches = 0;
    int max_check = std::min(10, M);
    for (int i = 0; i < max_check && mismatches < 5; i++) {
        for (int j = 0; j < max_check && mismatches < 5; j++) {
            float kernel_val = c[i*K + j];
            float cublas_val = c2[i*K + j];  // Same row-major access
            if (fabs(kernel_val - cublas_val) > eps) {
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << kernel_val << " != " << cublas_val << std::endl;
                mismatches++;
            }
        }
    }
    
    if (mismatches == 0) {
        std::cout << "✓ Results match within tolerance!" << std::endl;
    } else {
        std::cout << "⚠ Found " << mismatches << " mismatches in sample" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(c2);
    return 0;
}
