#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)

/******************************************************************************
 GPU kernels
*******************************************************************************/
#define uint unsigned int
__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the current level
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x ; idx < *numCurrLevelNodes ; idx += gridDim.x * blockDim.x)
  {
    uint node = currLevelNodes[idx];
    uint left = nodePtrs[node];
    uint right = nodePtrs[node + 1];

    // Loop over all neighbors of the node
    for(uint j = left ; j < right ; j ++)
    {
      // If neighbor hasn't been visited yet
      // Add neighbor to global queue
      uint neigh = nodeNeighbors[j];
      if(!atomicExch(&nodeVisited[neigh], 1))
      {
        uint qTop = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[qTop]= neigh;
      }
    }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  // Initialize shared memory queue (size should be BQ_CAPACITY)
  int idx = bx * blockDim.x + tx;
  __shared__ uint sharedQueue[BQ_CAPACITY]; 
  __shared__ uint numSharedQueue;
  __shared__ uint gqTop;
  if(tx == 0)
  {
    
    numSharedQueue = 0;
  }
  __syncthreads();
  
  // Loop over all nodes in the current level
  for(; idx < *numCurrLevelNodes ; idx += gridDim.x * blockDim.x)
  {
    // Loop over all neighbors of the node
    uint node = currLevelNodes[idx];
    uint left = nodePtrs[node];
    uint right = nodePtrs[node + 1];
    for(uint i = left ; i < right ; i ++)
    {
      uint neigh = nodeNeighbors[i];

      // If neighbor hasn't been visited yet
      if(!atomicExch(&nodeVisited[neigh], 1))
      {
        // If full, add neighbor to global queue
        uint bqTop = atomicAdd(&numSharedQueue, 1);
        if(bqTop >= BQ_CAPACITY)
        {
          nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neigh;
        }
        else
        {
          // Add neighbor to block queue
          sharedQueue[bqTop] = neigh;
        }
      }
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if(tx == 0)
    gqTop = atomicAdd(numNextLevelNodes, numSharedQueue);
  __syncthreads(); 

  // Store block queue in global queue
  for(int i = tx ; i < numSharedQueue; i += blockDim.x)
  {
    nextLevelNodes[gqTop + i] = sharedQueue[i];
  }
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  const int idx = bx * BLOCK_SIZE + tx;
  int laneId = tx % NUM_WARP_QUEUES;

  // This version uses NUM_WARP_QUEUES warp queues of capacity 
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.  
  __shared__ uint wQueue[WQ_CAPACITY][NUM_WARP_QUEUES]; 
  __shared__ uint numWQueue[NUM_WARP_QUEUES];

  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.
  __shared__ uint bQueue[BQ_CAPACITY];
  __shared__ uint numBQueue;

  // Initialize shared memory queues (warp and block)
  if(tx < NUM_WARP_QUEUES)
  {
    numWQueue[tx] = 0;
    if(tx == 0)
      numBQueue = 0;
  }
  __syncthreads();

  // Loop over all nodes in the current level
  for(int j = bx * BLOCK_SIZE + tx ; j < *numCurrLevelNodes ; j += gridDim.x * blockDim.x)
  {
    // printf("Hi i'm j: %d!\n", j);
    uint node = currLevelNodes[j];
    uint left = nodePtrs[node];
    uint right = nodePtrs[node + 1];

    // Loop over all neighbors of the node
    for(int i = left ; i < right ; i ++)
    {
      uint neigh = nodeNeighbors[i];
      // If neighbor hasn't been visited yet
      if(!atomicExch(&nodeVisited[neigh], 1))
      {
        // Add neighbor to the queue
        uint wqTop = atomicAdd(&numWQueue[laneId], 1);

        if(wqTop < WQ_CAPACITY)
        {
          wQueue[wqTop][laneId] = neigh;
        }
        else
        {
          atomicExch(&numWQueue[laneId], WQ_CAPACITY);
          uint bqTop = atomicAdd(&numBQueue, 1);

          // If full, add neighbor to block queue
          if(bqTop < BQ_CAPACITY)
          {
            bQueue[bqTop] = neigh;
          }
          else
          {
            // If full, add neighbor to global queue
            atomicExch(&numBQueue, BQ_CAPACITY);
            nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neigh;
          }
        } 
      }
    }
  }
  __syncthreads();

  if(tx < NUM_WARP_QUEUES)
  {
    uint num = numWQueue[laneId];
    // printf("Number of tx %d, bx %d, lane %d: %d\n", tx, bx, laneId, num);

    for(int i = 0 ; i < num ; i ++)
    {
      // Allocate space for warp queue to go into block queue
      uint bqTop = atomicAdd(&numBQueue, 1);
      uint val = wQueue[i][laneId];
      if(bqTop < BQ_CAPACITY) 
      {
        // Store warp queues in block queue (use one warp or one thread per queue)
        bQueue[bqTop] = val;
      }
      else
      {
        // Add any nodes that don't fit (remember, space was allocated above)
        // to the global queue
        nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = val;

        // Saturate block queue counter (too large if warp queues overflowed)
        atomicExch(&numBQueue, BQ_CAPACITY);
      }
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if(idx == 0)
  {
    // printf("WRITING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n");
    uint gqTop = atomicAdd(numNextLevelNodes, numBQueue);
    for(int i = 0 ; i < numBQueue ; i ++)
    {
      nextLevelNodes[gqTop + i] = bQueue[i];
    }
  }
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
