#include <cassert>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
// A is 16 x 16
// B is 16 x 8
// C is 16 x 8

#include <cuda.h>

template <typename T> struct Afrag_16x16 {
  static constexpr size_t ne = 8;

  T x[ne];

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2; // same as /4

    return group_id + 8 * ((l / 2) % 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return 2 * (tid % 4) + (l % 2) + 8 * (l / 4);
  }
};

// col major?
template <typename T> struct Bfrag_16x8 {
  static constexpr size_t ne = 4;
  T x[ne] = {};
  static __device__ size_t get_row(int tid, int l) {
    return (tid % 4) * 2 + (l % 2) + 8 * (l / 2);
  }

  static __device__ size_t get_col(int tid, int l) { return tid >> 2; }
};

template <typename T> struct CFrag_16x8 {
  static constexpr size_t ne = 4;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    return (tid >> 2) + 8 * (l / 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    assert(l < ne);
    return 2 * (tid % 4) + (l % 2);
  }
};

__global__ void mmaKernel(const half *A, const half *B, float *C, int M, int N,
                          int K) {
  Afrag_16x16<half> a_tile;
  Bfrag_16x8<half> b_tile;
  CFrag_16x8<float> c_tile;

  const int tid = threadIdx.x;

  __shared__ alignas(16) half A_shared[16][16 + 8];
  __shared__ alignas(16) half B_shared[16][8 + 8];

  const int lane = tid & 31;

  int c_row = blockIdx.y * 16;
  int c_col = blockIdx.x * 8;

  A += c_row * K;
  B += c_col;

  for (int k_idx = 0; k_idx < K; k_idx += 16) {

    if (lane < 16) {
      int row = lane;

      for (int idx = 0; idx < 8; ++idx) {
        A_shared[row][idx] = A[row * K + idx];
      }

      for (int idx = 0; idx < 8; ++idx) {
        A_shared[row][idx + 8] = A[row * K + 8 + idx];
      }

      for (int idx = 0; idx < 8; ++idx) {
        B_shared[row][idx] = B[row * N + idx];
      }
    }

    A += 16;
    B += 16 * N;

    int *a_regs = (int *)a_tile.x;
    int *b_regs = (int *)b_tile.x;

    int lane_id = tid;
    uint32_t a_addr =
        __cvta_generic_to_shared(&A_shared[(lane_id % 16)][(lane_id / 16) * 8]);
    uint32_t b_addr = __cvta_generic_to_shared(&B_shared[(lane_id % 16)]);

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]),
                   "=r"(a_regs[3])
                 : "r"(a_addr));

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                 "{%0, %1}, [%2];"
                 : "=r"(b_regs[0]), "=r"(b_regs[1])
                 : "r"(b_addr));

    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                   "+f"(c_tile.x[3])
                 : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),
                   "r"(a_regs[3]), "r"(b_regs[0]), "r"(b_regs[1]));
  }

  for (int i = 0; i < c_tile.ne; ++i) {
    int row = c_row + c_tile.get_row(tid, i);
    int col = c_col + c_tile.get_col(tid, i);

    C[row * N + col] = c_tile.x[i];
  }
}

__global__ void naiveKernel(const half *a, const half *b, float *c, int M,
                            int N, int K) {

  int row = blockIdx.y;
  int col = blockIdx.x;

  float tmp = 0;

  for (int i = 0; i < K; ++i) {
    tmp += (float)(a[row * K + i] * b[i * N + col]);
  }

  c[row * N + col] = tmp;
}

int main() {

  half *a;
  half *b;
  float *c;
  float *d;

  const int M = 1024;
  const int N = 1024;
  const int K = 32;

  cudaMallocManaged(&a, M * K * sizeof(half));
  cudaMallocManaged(&b, K * N * sizeof(half));
  cudaMallocManaged(&c, M * N * sizeof(float));
  cudaMallocManaged(&d, M * N * sizeof(float));

  for (int i = 0; i < M * K; ++i) {
    a[i] = __float2half((float)rand() / RAND_MAX);
  }

  for (int i = 0; i < K * N; ++i) {
    b[i] = __float2half((float)rand() / RAND_MAX);
  }

  // Benchmark mmaKernel vs cuBLAS
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const double flops = 2.0 * (double)M * (double)N * (double)K;

  // Time mmaKernel (Tensor Core MMA)
  cudaMemset(d, 0, M * N * sizeof(float));
  dim3 mma_grid(N / 8, M / 16);
  dim3 mma_block(32, 1, 1);
  // warmup
  mmaKernel<<<mma_grid, mma_block>>>(a, b, d, M, N, K);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  mmaKernel<<<mma_grid, mma_block>>>(a, b, d, M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float mma_ms = 0.0f;
  cudaEventElapsedTime(&mma_ms, start, stop);
  double mma_gflops = flops / (mma_ms * 1e6);

  // Time cuBLAS GemmEx (row-major via transposes: C^T = B^T * A^T)
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Use Tensor Op math if available
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaMemset(c, 0, M * N * sizeof(float));
  // warmup
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_16F,
               N,                // B^T (N x K) via row-major B (KxN)
               a, CUDA_R_16F, K, // A^T (K x M) via row-major A (MxK)
               &beta, c, CUDA_R_32F,
               N, // C^T (N x M) stored in row-major C (MxN)
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(start);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_16F,
               N, a, CUDA_R_16F, K, &beta, c, CUDA_R_32F, N, CUDA_R_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float cublas_ms = 0.0f;
  cudaEventElapsedTime(&cublas_ms, start, stop);
  double cublas_gflops = flops / (cublas_ms * 1e6);

  // Report
  printf("\n=== GEMM Benchmark (M=%d, N=%d, K=%d) ===\n", M, N, K);
  printf("FLOPs: %.3f GFLOPs\n", flops * 1e-9);
  printf("mmaKernel:  %.3f ms  |  %.2f GFLOPs\n", mma_ms, mma_gflops);
  printf("cuBLAS:     %.3f ms  |  %.2f GFLOPs\n", cublas_ms, cublas_gflops);
  printf("Speedup (cuBLAS / mma): %.2fx\n", mma_ms / cublas_ms);

  int mismatches = 0;
  const float eps = 1e-2f;
  for (int i = 0; i < 10 && i < M; ++i) {
    for (int j = 0; j < 10 && j < N; ++j) {
      float v_mma = d[i * N + j];
      float v_blas = c[i * N + j];
      if (fabsf(v_mma - v_blas) > eps) {
        ++mismatches;
        if (mismatches > 5)
          break;
      }
    }
    if (mismatches > 5)
      break;
  }
  if (mismatches == 0) {
    printf("Results match within tolerance (eps=%.1e)\n", eps);
  } else {
    printf("Found %d mismatches in 10x10 spot-check (eps=%.1e)\n", mismatches,
           eps);
  }

  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
