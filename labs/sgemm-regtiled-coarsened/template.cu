#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define T 128
#define U 16
#define S (T / U)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row = bx * T + tx;
  int col = by * U;
  float tmpC[U] = {0.};

  __shared__ float tmpB[S][U];

  for(int offset = 0 ; offset < k ; offset+= S)
  {
    int x = tx % U;
    int y = tx / U;
    tmpB[y][x] = (col + x < n && offset + y < k) ? B(offset + y, col + x) : 0.;

    __syncthreads();

    for(int i = 0 ; i < S ; i ++)
    {
        if(row < m && offset + i < k)
        {
            float valA = A(row, offset + i); 
            for(int u = 0 ; u < U ; u ++)
            {
                tmpC[u] += valA * tmpB[i][u];
            }
        }
    }
    __syncthreads();
  }

  if(row < m)
  {
    for(int u = 0 ; u < U ; u ++)
    {
        C(row, col + u) = tmpC[u];
    }
  }

  #undef A
  #undef B
  #undef C
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    // Your code need only consider the m, n, k, A, B, and C parameters of
    // the function, which provide the matrix sizes (m, n, k) and data
    // (A, B, C).

    //INSERT CODE HERE
    dim3 dimGrid(ceil(1.0 * m / T), ceil(1.0 * n / U));
    dim3 dimBlock(T);

    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);

    //INSERT CODE HERE

}

