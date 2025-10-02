#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>

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

  __shared__ alignas(16) half A_shared[16][16];
  __shared__ alignas(16) half B_shared[16][8];

  const int lane = tid & 31;

  if (lane < 16) {
    int row = lane;

    for(int idx = 0; idx < 8; ++idx) {
        A_shared[row][idx] = A[row*K + idx];
    }

    for(int idx = 0; idx < 8; ++idx) {
        A_shared[row][idx + 8] = A[row*K + 8 + idx];
    }

    for(int idx = 0 ; idx < 8; ++idx) {
        B_shared[row][idx] = B[row*N + idx];
    }
  }

  int *a_regs = (int *)a_tile.x;
  int *b_regs = (int *)b_tile.x;

  int lane_id = tid;
  uint32_t a_addr = __cvta_generic_to_shared(
      &A_shared[(lane_id % 16)][(lane_id / 16) * 8]);
  uint32_t b_addr = __cvta_generic_to_shared(
      &B_shared[(lane_id % 16)]);

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
               : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                 "r"(b_regs[0]), "r"(b_regs[1]));


  for (int i = 0; i < c_tile.ne; ++i) {
    int row = c_tile.get_row(tid, i);
    int col = c_tile.get_col(tid, i);

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

  const int M = 16;
  const int N = 8;
  const int K = 16;

  cudaMallocManaged(&a, M * K * sizeof(half));
  cudaMallocManaged(&b, K * N * sizeof(half));
  cudaMallocManaged(&c, M * N * sizeof(float));
  cudaMallocManaged(&d, M * N * sizeof(float));

  for (int i = 0; i < 16 * 16; ++i) {
    a[i] = __float2half((float)rand() / RAND_MAX);
  }

  for (int i = 0; i < 16 * 8; ++i) {
    b[i] = __float2half((float)rand() / RAND_MAX);
  }

  dim3 grid(N, M);
  dim3 block(1, 1, 1);

  naiveKernel<<<grid, block>>>(a, b, c, M, N, K);

  dim3 mma_grid(1, 1);
  dim3 mma_block(32, 1, 1);
  mmaKernel<<<mma_grid, mma_block>>>(a, b, d, M, N, K);

  cudaDeviceSynchronize();

  float rmse = 0.;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float err = std::abs(d[i * N + j] - c[i * N + j]);
      if (err > 1e-3) {
        printf("Mismatched [%d, %d]: err: %.2f %.5f %.5f \n", i, j, err,
               d[i * N + j], c[i * N + j]);
      }
      rmse += err * err;
    }
  }

  printf("RMSE: %.2f", sqrt(rmse));
}
