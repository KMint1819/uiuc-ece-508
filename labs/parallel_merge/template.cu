#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
  return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
  int i = 0, j = 0, k = 0;

  while ((i < A_len) && (j < B_len)) {
    C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
  }

  if (i == A_len) {
    while (j < B_len) {
      C[k++] = B[j++];
    }
  } else {
    while (i < A_len) {
      C[k++] = A[i++];
    }
  }
}

__device__ int co_rank(int k, float* A, int m, float* B, int n) {
  int low = max(k - n, 0);
  int high = (k < m ? k : m);
  while (low < high) {
    int i = low + (high - low) / 2;
    int j = k - i;
    if (i > 0 && j < n && A[i - 1] > B[j]) {
      high = i - 1;
    } else if (j > 0 && i < m && A[i] <= B[j - 1]) {
      low = i + 1;
    } else {
      return i;
    }
  }
  return low;
}
/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C_len = A_len + B_len;
    int ept = ceil_div(C_len, blockDim.x * gridDim.x);
    
    int k_curr = min(idx * ept, C_len);
    int k_next = min(k_curr + ept, C_len);

    int i_curr = co_rank(k_curr, A, A_len, B, B_len);
    int i_next = co_rank(k_next, A, A_len, B, B_len);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
  /* Your code here */
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
  /* Your code here */
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = 128;
  gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = 128;
  gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = 128;
  gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
