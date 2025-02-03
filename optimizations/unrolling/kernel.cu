// Unrolling: Unroll matrices for convolution to reduce memory access latency.

#include <cmath>
#include <iostream>
#include "gpu.h"

#define TILE_WIDTH 16
__constant__ float const_mask[16384];

__global__ void unroll_kernel(int C, int H, int W, int K, float* X, float* X_unroll) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if(t < C * W_unroll) {
        int c = t/W_unroll;
        int w_unroll = t % W_unroll;
        int h_out = w_unroll / W_out;
        int w_out = w_unroll % W_out;
        int w_base = c * K * K;
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q;
                X_unroll[h_unroll * W_unroll + w_unroll] = X[c * H * W + (h_out + p) * W + (w_out + q)];
            }
        }
    }  
}

__global__ void matrixMultiplyShared(float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

  __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float temp = 0;
  for(int m = 0; m < (numAColumns - 1)/TILE_WIDTH + 1; m++) {
    // load matrices from device to shared memory
    if(row < numARows && m * TILE_WIDTH + tx < numAColumns) {
      A_s[ty][tx] = const_mask[row * numAColumns + m * TILE_WIDTH + tx];
    } else {
      A_s[ty][tx] = 0;
    }

    if(col < numBColumns && m * TILE_WIDTH + ty < numBRows) {
      B_s[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + col];
    } else {
      B_s[ty][tx] = 0;
    }
    __syncthreads();

    // multiply shared matrices together
    for(int k = 0; k < TILE_WIDTH; k++) {
      temp += A_s[ty][k] * B_s[k][tx];
    }
    __syncthreads();
  }
  //save result to output matrix
  if(row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = temp;
  }
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory for GPU
    cudaMalloc((void**) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void**) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    
    // and copy over the relevant data structures to the GPU
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_mask, host_mask, Map_out * Channel * K * K * sizeof(float));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // set dimensions for matrix multiplication kernel
    dim3 dimGrid_matmul(ceil(float(Height_out * Width_out) / TILE_WIDTH), ceil(float(Map_out) / TILE_WIDTH), 1);

    // Initialize and allocate unrolled input
    float* X_unroll;
    cudaMalloc((void **) &X_unroll, Channel * K * K * Height_out * Width_out * sizeof(float)); // reusable buffer for unrolled input per image
    for(int i = 0; i < Batch; i++) { // for each image in the batch
    //unroll_kernel(int C, int H, int W, int K, float* X, float* X_unroll)
      unroll_kernel<<<ceil(float(Channel * Height_out * Width_out) / 1024), 1024>>>(Channel, Height, Width, K, (float *)device_input + i * Channel * Height * Width, X_unroll); // unroll the image
      cudaDeviceSynchronize();
      // then multiply the mask by the unrolled input
      matrixMultiplyShared<<<dimGrid_matmul,blockDim>>>(X_unroll, device_output + i * Map_out * Height_out * Width_out, Map_out, Channel * K * K, Channel * K * K, Height_out * Width_out, Map_out, Height_out * Width_out);
      cudaDeviceSynchronize();
    }
    cudaFree(X_unroll);
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