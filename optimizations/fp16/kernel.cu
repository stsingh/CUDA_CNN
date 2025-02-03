// FP16: Convert input and mask to fp16 to reduce memory bandwidth usage and increase performance

#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu.h"

#define TILE_WIDTH 16
__global__ void fp16_convert(const float *input, const float *mask, __half* fp16_input, __half* fp16_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
	#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	#define fp16_in_4d(i3, i2, i1, i0) fp16_input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define fp16_mask_4d(i3, i2, i1, i0) fp16_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int Width_out = Width - K + 1;
    int W_grid = (Width_out + TILE_WIDTH - 1)/TILE_WIDTH; // get width in terms of tiles

	int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

	if(n < Batch && h < Height && w < Width) {
		for(int c = 0; c < Channel; c++) {
			fp16_in_4d(n, c, h, w) = __float2half_rn(in_4d(n, c, h, w));
		}
	}

	if(m < Map_out && w < K && h < K) { // since no p and q, reuse w and k as long as they're in range
		for(int c = 0; c < Channel; c++) {
			fp16_mask_4d(m, c, h, w) = __float2half_rn(mask_4d(m, c, h, w));
		}
	}
    #undef in_4d
    #undef mask_4d
	#undef fp16_in_4d
    #undef fp16_mask_4d
}



__global__ void conv_forward_kernel(float *output, const __half *input, __half *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = (Width_out + TILE_WIDTH - 1)/TILE_WIDTH; // get width in terms of tiles

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    __half pix_sum = 0.0;
    if(h < Height_out && w < Width_out) {
        for (c = 0; c < Channel; c++) {
            for (p = 0; p < K; p++) {
                for (q = 0; q < K; q++) {
                    pix_sum = __hadd(pix_sum, in_4d(n, c, h + p, w + q) * mask_4d(m, c, p, q));
                }
            }
        }
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
    // Set the kernel dimensions for original kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int H_grid = (Height_out + (TILE_WIDTH - 1)) / TILE_WIDTH; // Get height in terms of tiles
    int W_grid = (Width_out + (TILE_WIDTH - 1))/TILE_WIDTH; // get width in terms of tiles

	// malloc __half buffers
	__half* fp16_d_in;
	__half* fp16_d_mask;
	cudaMalloc((void **)&fp16_d_in, Batch * Channel * Height * Width * sizeof(__half));
	cudaMalloc((void **)&fp16_d_mask, Map_out * Channel * K * K * sizeof(__half));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Batch, Map_out, H_grid * W_grid);
	fp16_convert<<<gridDim, blockDim>>>(device_input, device_mask, fp16_d_in, fp16_d_mask, Batch, Map_out, Channel, Height, Width, K);
	cudaDeviceSynchronize();
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, fp16_d_in, fp16_d_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();

	cudaFree(fp16_d_in);
	cudaFree(fp16_d_mask);
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