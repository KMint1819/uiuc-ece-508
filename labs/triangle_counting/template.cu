#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__device__ uint64_t linear_search(
  const uint32_t* const edgeDst,
  int u,
  int v,
  int uEnd,
  int vEnd 
)
{
  uint64_t ans = 0;

  // Determine how many elements of those two arrays are common
  int w1 = edgeDst[u];
  int w2 = edgeDst[v];
  while(u < uEnd && v < vEnd)
  {
    if(w1 > w2)
    {
      w2 = edgeDst[++ v];
    }
    else if(w1 < w2)
    {
      w1 = edgeDst[++ u];
    }
    else
    {
      w1 = edgeDst[++ u];
      w2 = edgeDst[++ v];
      ans ++;
    }
  }
  return ans;
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
  // Determine the source and destination node for the edge
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(idx >= numEdges) 
    return;

  int src = edgeSrc[idx];
  int dst = edgeDst[idx];

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  int u = rowPtr[src];
  int v = rowPtr[dst];
  int uEnd = rowPtr[src + 1];
  int vEnd = rowPtr[dst + 1];

  triangleCounts[idx] = linear_search(edgeDst, u, v, uEnd, vEnd);
}

__device__ uint64_t binary_search(
  const uint32_t* const edgeDst,
  int u,
  int v,
  int uEnd,
  int vEnd 
){
  uint64_t ans = 0;
  for(; u < uEnd ; u ++)
  {
    int target = edgeDst[u];
    int l = v;
    int r = vEnd - 1;
    while(l <= r)
    {
      int mid = (l + r) / 2;
      int val = edgeDst[mid];
      if(target > val)
      {
        l = mid + 1;
      }
      else if(target < val)
      {
        r = mid - 1;
      }
      else
      {
        ans ++;
        break; 
      }
    }
  }
  return ans;
}

__device__ void swap_integer(int& a, int& b)
{
  int tmp = a;
  a = b;
  b = tmp;
}

__global__ static void kernel_tc_bs(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= numEdges) return;

  int src = edgeSrc[idx];
  int dst = edgeDst[idx];

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  int u = rowPtr[src];
  int v = rowPtr[dst];
  int uEnd = rowPtr[src + 1];
  int vEnd = rowPtr[dst + 1];
  
  if(uEnd - u > vEnd - v)
  {
    swap_integer(u, v);
    swap_integer(uEnd, vEnd);
  }
  // if(v >= 64 && )
  triangleCounts[idx] = binary_search(edgeDst, u, v, uEnd, vEnd);
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
  pangolin::Vector<uint64_t> pgl(view.nnz(), 0);


  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  dim3 dimGrid (ceil(1.0 * view.nnz() / dimBlock.x));

  if (mode == 1) 
  {
    //@@ launch the linear search kernel here
    kernel_tc<<<dimGrid, dimBlock>>>(pgl.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());

  } 
  else if (mode == 2) 
  {
    //@@ launch the hybrid search kernel here
    kernel_tc_bs<<<dimGrid, dimBlock>>>(pgl.data(),view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    cudaDeviceSynchronize();
  }
  else 
  {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  cudaDeviceSynchronize();
  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  uint64_t total = 0;
  for(int i = 0 ; i < view.nnz() ; i ++)
  {
    total += pgl[i];
  }
  return total;
}
