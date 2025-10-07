#include <cassert>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>

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

struct params {

  static const int BM = 128;
  static const int BN = 64;
  static const int BK = 32;

  static const int WM = 64;
  static const int WN = 32;
  static const int WK = 32;

  static const int A_row_stride = BK + 8;
  static const int B_row_stride = BN + 8;

  static const int MMA_M = 16;
  static const int MMA_N = 8;
  static const int MMA_K = 16;

  static const int SM =
      (BM * A_row_stride) * sizeof(half) + (BK * B_row_stride) * sizeof(half);
};

__global__ void mmaKernel2(const half *__restrict__ A,
                           const half *__restrict__ B, float *__restrict__ C,
                           int M, int N, int K) {

  using tileA = Afrag_16x16<half>;
  using tileB = Bfrag_16x8<half>;
  using tileC = CFrag_16x8<float>;

  const int c_row = blockIdx.y * params::BM;
  const int c_col = blockIdx.x * params::BN;

  // Move start_a rows down
  A += c_row * K;
  B += c_col;

  const int warp_idx = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const int n_warps = blockDim.x / 32;

  const int warp_row = warp_idx / 2;
  const int warp_col = warp_idx % 2;

  extern __shared__ unsigned char data[];

  const int WM_TILES = params::WM / params::MMA_M;
  const int WN_TILES = params::WN / params::MMA_N;
  const int WK_TILES = params::WK / params::MMA_K;

  // assert(WM_TILES == 2);
  // assert(WN_TILES == 4);
  // assert(WK_TILES == 2);
  tileC c_tiles[WM_TILES][WN_TILES];

  half *A_shared = (half *)data;
  half *B_shared = (half *)data + (params::BM * (params::A_row_stride));

#pragma unroll
  for (int row_idx = warp_idx; row_idx < params::BM; row_idx += n_warps) {
    // load 8 floats at a time
    if (lane_id < params::BK / 8) {
      const uint4 *ptr =
          reinterpret_cast<const uint4 *>(&A[row_idx * K + lane_id * 8]);
      uint4 *write_ptr = reinterpret_cast<uint4 *>(
          &A_shared[row_idx * params::A_row_stride + lane_id * 8]);
      *write_ptr = *ptr;
    }
    if (lane_id < params::BN / 8 && row_idx < params::BK) {
      const uint4 *ptr =
          reinterpret_cast<const uint4 *>(&B[row_idx * N + lane_id * 8]);
      uint4 *write_ptr = reinterpret_cast<uint4 *>(
          &B_shared[row_idx * params::B_row_stride + lane_id * 8]);
      *write_ptr = *ptr;
    }
  }

  __syncthreads();

  int read_index = 0;
  int write_index = 1;

  for (int k_tile = params::BK; k_tile <= K; k_tile += params::BK) {
    unsigned char *write_data = data + write_index * params::SM;
    unsigned char *read_data = data + read_index * params::SM;
    A_shared = (half *)write_data;
    B_shared = (half *)write_data + (params::BM * (params::A_row_stride));

    half *A_shared_read = (half *)read_data;
    half *B_shared_read =
        (half *)read_data + (params::BM * (params::A_row_stride));

    if (k_tile < K) {
      A += params::BK;     // move columns
      B += params::BK * N; // move BK columns down

#pragma unroll
      for (int row_idx = warp_idx; row_idx < params::BM; row_idx += n_warps) {
        // load 8 floats at a time
        if (lane_id < params::BK / 8) {
          const uint4 *ptr =
              reinterpret_cast<const uint4 *>(&A[row_idx * K + lane_id * 8]);
          uint4 *write_ptr = reinterpret_cast<uint4 *>(
              &A_shared[row_idx * params::A_row_stride + lane_id * 8]);
          *write_ptr = *ptr;
        }
        if (lane_id < params::BN / 8 && row_idx < params::BK) {
          const uint4 *ptr =
              reinterpret_cast<const uint4 *>(&B[row_idx * N + lane_id * 8]);
          uint4 *write_ptr = reinterpret_cast<uint4 *>(
              &B_shared[row_idx * params::B_row_stride + lane_id * 8]);
          *write_ptr = *ptr;
        }
      }
    }

    // move according to your warp
    half *a_shared_ptr =
        A_shared_read + (warp_row * params::WM) * params::A_row_stride;
    half *b_shared_ptr = B_shared_read + warp_col * params::WN;

#pragma unroll
    for (int mma_k = 0; mma_k < WK_TILES; mma_k++) {
#pragma unroll
      for (int mma_m = 0; mma_m < WM_TILES; mma_m++) {
        tileA tile_a;
        half *a_ptr = a_shared_ptr +
                      (mma_m * params::MMA_M) * params::A_row_stride +
                      mma_k * params::MMA_K;
        const int row = lane_id % 16;
        const int col = (lane_id / 16) * 8;
        const int row_stride = params::A_row_stride;
        a_ptr += row * row_stride + col;

        int *a_regs = (int *)tile_a.x;

        uint32_t a_addr = __cvta_generic_to_shared(a_ptr);

        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]),
                       "=r"(a_regs[3])
                     : "r"(a_addr));

#pragma unroll
        for (int mma_n = 0; mma_n < WN_TILES; mma_n++) {
          // tileB

          tileB tileB;

          half *b_ptr = b_shared_ptr +
                        (mma_k * params::MMA_K) * params::B_row_stride +
                        mma_n * params::MMA_N;

          tileC &c_tile = c_tiles[mma_m][mma_n];

          const int b_row = lane_id % 16;
          const int b_row_stride = params::B_row_stride;
          b_ptr += b_row * b_row_stride;

          int *b_regs = (int *)tileB.x;

          uint32_t b_addr = __cvta_generic_to_shared(b_ptr);
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                       "{%0, %1}, [%2];"
                       : "=r"(b_regs[0]), "=r"(b_regs[1])
                       : "r"(b_addr));

          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                       "{%0, %1, %2, %3}, "
                       "{%4, %5, %6, %7}, "
                       "{%8, %9}, "
                       "{%0, %1, %2, %3};\n"
                       : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]),
                         "+f"(c_tile.x[2]), "+f"(c_tile.x[3])
                       : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),
                         "r"(a_regs[3]), "r"(b_regs[0]), "r"(b_regs[1]));
        }
      }
    }
    __syncthreads();
    read_index ^= 1;
    write_index ^= 1;
  }

  float *c_shared = (float *)data;

  // epilogue
  // load up BM * BN ito shared mem
  float *c_ptr =
      c_shared + (warp_row * params::WM) * params::BN + warp_col * params::WN;

#pragma unroll
  for (int mma_m = 0; mma_m < WM_TILES; mma_m++) {
#pragma unroll
    for (int mma_n = 0; mma_n < WN_TILES; mma_n++) {
      tileC &c_tile = c_tiles[mma_m][mma_n];

#pragma unroll
      for (int n = 0; n < c_tile.ne; n++) {
        const int r = (mma_m * params::MMA_M) + tileC::get_row(lane_id, n);
        const int c = (mma_n * params::MMA_N) + tileC::get_col(lane_id, n);
        c_ptr[r * params::BN + c] = c_tile.x[n];
      }
    }
  }

  __syncthreads();

#pragma unroll
  for (int row_idx = warp_idx; row_idx < params::BM; row_idx += n_warps) {
    if (lane_id < params::BN / 4) {
      float4 *ptr = (float4 *)&c_shared[row_idx * params::BN + lane_id * 4];
      float4 *global_ptr =
          (float4 *)&C[(c_row + row_idx) * N + c_col + lane_id * 4];
      *global_ptr = *ptr;
    }
  }
}

__global__ void flushL2Cache(float *flush_buf, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // Write to buffer to evict L2 cache lines
  for (size_t i = idx; i < size; i += stride) {
    flush_buf[i] = (float)i;
  }
}

int main() {

  half *a;
  half *b;
  float *c;
  float *d;

  const int M = 8192;
  const int N = 8192;
  const int K = 8192;

  cudaMallocManaged(&a, M * K * sizeof(half));
  cudaMallocManaged(&b, K * N * sizeof(half));
  cudaMallocManaged(&c, M * N * sizeof(float));
  cudaMallocManaged(&d, M * N * sizeof(float));

  // Allocate buffer to flush L2 cache
  // RTX 3090 has 6MB L2 cache, allocate larger buffer to ensure full flush
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  size_t l2_cache_size = prop.l2CacheSize;
  size_t flush_size =
      (l2_cache_size * 2) / sizeof(float); // 2x L2 size in floats

  float *flush_buf;
  cudaMalloc(&flush_buf, flush_size * sizeof(float));

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

  const int nbytes_a = (params::BM * (params::BK + 8) * sizeof(half));
  const int nbytes_b = (params::BK * (params::BN + 8) * sizeof(half));

  const int nbytes_c = (params::BM * params::BN) * sizeof(float);

  printf("Total smem per block with double buffer %d, for C: %d\n",
         2 * (nbytes_a + nbytes_b), nbytes_c);

  // Query device for shared memory capabilities
  device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  int max_shared_per_block = deviceProp.sharedMemPerBlock;
  int max_shared_optin = 0;
  cudaDeviceGetAttribute(&max_shared_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

  printf("\n=== Shared Memory Info ===\n");
  printf("Default max shared memory per block: %d KB\n",
         max_shared_per_block / 1024);
  printf("Opt-in max shared memory per block: %d KB\n",
         max_shared_optin / 1024);

  // Calculate required shared memory
  const int nbytes_shared = std::max(2 * (nbytes_a + nbytes_b), nbytes_c);
  printf("Required shared memory: %d KB\n", nbytes_shared / 1024);

  // Request increased shared memory limit if needed
  if (nbytes_shared > max_shared_per_block &&
      nbytes_shared <= max_shared_optin) {
    printf("Requesting increased shared memory limit...\n");
    cudaFuncSetAttribute(
        mmaKernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);

    cudaFuncSetAttribute(mmaKernel2,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
  }

  if (nbytes_shared > max_shared_optin) {
    printf(
        "ERROR: Required shared memory (%d KB) exceeds device limit (%d KB)!\n",
        nbytes_shared / 1024, max_shared_optin / 1024);
    printf("Consider reducing BM, BN, or BK.\n");
    return -1;
  }

  cudaMemset(d, 0, M * N * sizeof(float));
  dim3 mma_grid(N / params::BN, M / params::BM);

  int n_warps = (params::BM / params::WM) * (params::BN / params::WN);
  dim3 mma_block(n_warps * 32, 1, 1);

  // warmup
  mmaKernel2<<<mma_grid, mma_block, nbytes_shared>>>(a, b, d, M, N, K);
  // check error
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  // Flush L2 cache before timed run
  dim3 flush_grid(1024);
  dim3 flush_block(256);
  flushL2Cache<<<flush_grid, flush_block>>>(flush_buf, flush_size);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  mmaKernel2<<<mma_grid, mma_block, nbytes_shared>>>(a, b, d, M, N, K);
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

  cudaDeviceSynchronize();

  // Flush L2 cache before timed run
  flushL2Cache<<<flush_grid, flush_block>>>(flush_buf, flush_size);
  cudaDeviceSynchronize();

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
  cudaFree(flush_buf);
}
