// Shared memory: use the shared memory in a thread block to optimize bandwidth usage.

#include <cmath>
#include <iostream>
#include "gpu.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = (Width_out + TILE_WIDTH - 1)/TILE_WIDTH; // get width in terms of tiles

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int n, m, h, w, c, p, q;
    int X_tile_width = TILE_WIDTH + K - 1; // Width of input tile
    extern __shared__ float shmem[]; // shared memory (defined in kernel call)
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];
    
    // define program variables
    n = blockIdx.x;
    m = blockIdx.y;
    int h0 = threadIdx.y; int w0 = threadIdx.x;
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;
    float pix_sum = 0.0;

    for(c = 0; c < Channel; c++) {
      // load mask into shmem
      if(w0 < K && h0 < K) {
        W_shared[h0 * K + w0] = mask_4d(m, c, h0, w0);
      }
      __syncthreads();

      // load tile of input into shmem
      for(int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
        for(int j = w; j < w_base + X_tile_width; j+= TILE_WIDTH) {
          if(i < Height && j < Width) {
            X_shared[(i - h_base) * X_tile_width + (j - w_base)] = in_4d(n, c, i, j);
          } else {
            X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0.0;
          }
        }
      }
      __syncthreads();

      // partial sum
      for(p = 0; p < K; p++) {
        for(q = 0; q < K; q++) {
          if(h0 + p < X_tile_width && w0 + q < X_tile_width) {
            pix_sum += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
          }
        }
      }
      __syncthreads();
    }

    // write to output
    if(n < Batch && m < Map_out && h < Height_out && w < Width_out) {
      out_4d(n, m, h, w) = pix_sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory for GPU
    cudaMalloc((void**) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void**) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    
    // and copy over the relevant data structures to the GPU
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int H_grid = (Height_out + (TILE_WIDTH - 1)) / TILE_WIDTH; // Get height in terms of tiles
    int W_grid = (Width_out + (TILE_WIDTH - 1))/TILE_WIDTH; // get width in terms of tiles
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Batch, Map_out, H_grid * W_grid);
    size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
    conv_forward_kernel<<<gridDim, blockDim, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}