#include "helper.hpp"

#define TILE_WIDTH 8
#define MASK_NUM 32
#define MASK_WIDTH 5
__constant__ float mask[MASK_NUM * MASK_WIDTH * MASK_WIDTH];

// Sequential code for the forward path of the convolution layer
// You should not modify this code
// static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
//                                const shape &ydims) {
//   std::fill(Y, Y + ydims.flattened_length(), 0);

//   for (auto i : range(0, ydims.num)) {
//     for (auto m : range(0, ydims.depth )) {   // for each output feature map
//       for (auto h : range(0, ydims.height)) { // for each output element
//         for (auto w : range(0, ydims.width )) {
//           const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
//           for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
//             for (auto p : range(0, wdims.height)) {   // filter height
//               for (auto q : range(0, wdims.width )) { // filter width
//                 const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
//                 const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
//                 Y[yoffset] += X[xoffset] * W[woffset];
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
                                    const shape ydims) {


  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth )) { // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
  const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());

}

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(
  const float *x, 
  const shape xdims, 
  const float *_, 
  const shape wdims, 
  float *y,
  const shape ydims) 
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //@@ YOUR CODE HERE!

  int wgrid = ceil(1.0 * ydims.width / TILE_WIDTH);
  int b = bz;
  int m = bx;
  int h = (by / wgrid) * TILE_WIDTH + ty;
  int w = (by % wgrid) * TILE_WIDTH + tx;
  int radius = TILE_WIDTH / 2;

  #define X(i0, i1) x[((b)) * xdims.height * xdims.width + (i0) * xdims.width + (i1)]
  #define M(i0, i1) mask[((m)) * wdims.height * wdims.width + (i0) * wdims.width + (i1)]
  #define Y(i0, i1, i2, i3) y[((i0) * ydims.depth + (i1)) * ydims.height * ydims.width + (i2) * ydims.width + (i3)]

  __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];
  if(h < xdims.height && w < xdims.width)
  {
    sharedX[ty][tx] = X(h, w);
  }
  else
  {
    sharedX[ty][tx] = 0.0f;
  }
  __syncthreads();

  float ans = 0.0f;
  for(int p = 0 ; p < MASK_WIDTH ; p ++)
  {
    for(int q = 0 ; q < MASK_WIDTH ; q ++)
    {
      if(h + p < xdims.height && w + q < xdims.width)
      {
        if(ty + p < radius || tx + q < radius || ty + p >= TILE_WIDTH - radius || tx + q >= TILE_WIDTH - radius)
        {
          ans += X(h + p, w + q) * M(p, q);
        }
        else
        {
          ans += sharedX[ty + p][tx + q] * M(p, q);
        }
      }
    }
  }
  if(h < ydims.height && w < ydims.width)
  {
    Y(b, m, h, w) = ans;
  }
  #undef X
  #undef W
  #undef Y
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(const float *X, 
  const shape &xdims, 
  const float *W, 
  const shape &wdims, 
  float *Y, 
  const shape &ydims) {

  // Modify this code to configure your optimized kernel.
  //@@ YOUR CODE HERE!!!
  cudaMemcpyToSymbol(mask, W, MASK_NUM * MASK_WIDTH * MASK_WIDTH * sizeof(float), 0, cudaMemcpyHostToDevice);

  int hgrid = ceil(1.0 * ydims.height / TILE_WIDTH);
  int wgrid = ceil(1.0 * ydims.width / TILE_WIDTH);

  dim3 dimGrid(ydims.depth, hgrid * wgrid, xdims.num);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  conv_forward_opt_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  cudaDeviceSynchronize();
  THROW_IF_ERROR(cudaGetLastError());
}


static int eval(const shape wDims, const shape xDims, bool doVerify) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," +
                                                      std::to_string(wDims.depth) + "," +
                                                      std::to_string(wDims.height) + "," +
                                                      std::to_string(wDims.width) +
                                                      " xDims:" + std::to_string(xDims.num) + "," +
                                                      std::to_string(xDims.depth) + "," +
                                                      std::to_string(xDims.height) + "," +
                                                      std::to_string(xDims.width) + "]";
  INFO("Running "  << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1),
      (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);


  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **)&deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceY, yByteCount));
  timer_stop();


  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  if (doVerify) {
    timer_start("Copying output to the CPU");
    THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
    timer_stop();

    convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
    // conv_forward_valid(hostX, xDims, hostW, wDims, expected, ydims);
    verify(expected, hostY, ydims);
  }

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}



TEST_CASE("Convlayer", "[convlayer]") 
{
#if 1
  // test five times in case code errors depend on data
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
#else
  SECTION("[wDims:32,1,5,5 xDims:50000,1,28,28]") {
    eval({32,1,5,5}, {50000,1,28,28}, false);
  }
#endif
}
