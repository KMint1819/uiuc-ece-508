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
  int low  = max(k - n, 0);
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
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int C_len = A_len + B_len;
  int ept   = ceil_div(C_len, blockDim.x * gridDim.x);

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
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    __shared__ float tile[TILE_SIZE * 2];
    float *tileA = &tile[0];
    float *tileB = &tile[TILE_SIZE];

    int elt = ceil_div(A_len + B_len, gridDim.x);
    int blk_k_curr = min(bx * elt, A_len + B_len);
    int blk_k_next = min(blk_k_curr + elt, A_len + B_len);

    if(tx == 0)
        tile[0] = co_rank(blk_k_curr, A, A_len, B, B_len);
    if(tx == 1)
        tile[1] = co_rank(blk_k_next, A, A_len, B, B_len);
    __syncthreads();

    // Define some variables
    int blk_i_curr = tile[0];
    int blk_i_next = tile[1];
    int blk_j_curr = blk_k_curr - blk_i_curr;
    int blk_j_next = blk_k_next - blk_i_next;
    int A_length = blk_i_next - blk_i_curr;
    int B_length = blk_j_next - blk_j_curr;
    int C_length = blk_k_next - blk_k_curr;
    int num_tiles = ceil_div(C_length, TILE_SIZE);
    int A_consumed = 0;
    int B_consumed = 0;
    int C_produced = 0;

    for(int counter = 0 ; counter < num_tiles ; counter ++) {
        __syncthreads();
        int valid_A_length = min(A_length - A_consumed, TILE_SIZE);
        int valid_B_length = min(B_length - B_consumed, TILE_SIZE);
        int valid_C_length = min(C_length - C_produced, TILE_SIZE);
        for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
            if (i + tx < valid_A_length)
                tileA[i + tx] = A[blk_i_curr + A_consumed + i + tx];
            if (i + tx < valid_B_length)
                tileB[i + tx] = B[blk_j_curr + B_consumed + i + tx];
        }
        __syncthreads();

        int per_thread = ceil_div(TILE_SIZE, blockDim.x);
        int thd_k_curr = min(per_thread * tx, valid_C_length);
        int thd_k_next = min(thd_k_curr + per_thread, valid_C_length);

        int thd_i_curr = co_rank(thd_k_curr, tileA, valid_A_length, tileB, valid_B_length);
        int thd_i_next = co_rank(thd_k_next, tileA, valid_A_length, tileB, valid_B_length);
        int thd_j_curr = thd_k_curr - thd_i_curr;
        int thd_j_next = thd_k_next - thd_i_next;

        // merge
        merge_sequential(&tileA[thd_i_curr], thd_i_next - thd_i_curr, &tileB[thd_j_curr],
                        thd_j_next - thd_j_curr, &C[blk_k_curr + C_produced + thd_k_curr]);

        A_consumed += co_rank(valid_C_length, tileA, valid_A_length, tileB, valid_B_length);
        C_produced += valid_C_length;
        B_consumed = C_produced - A_consumed;
    }
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
