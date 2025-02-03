// Streams: Overlap kernel calls with data transfer to induce speedup.

#include <cmath>
#include <iostream>
#include "gpu.h"

#define TILE_WIDTH 16
#define num_streams 10

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = (Width_out + TILE_WIDTH - 1)/TILE_WIDTH; // get width in terms of tiles

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    float pix_sum = 0.0;
    if(h < Height_out && w < Width_out) {
        for (c = 0; c < Channel; c++) {
            for (p = 0; p < K; p++) {
                for (q = 0; q < K; q++) {
                    pix_sum += in_4d(n, c, h + p, w + q) * mask_4d(m, c, p, q);
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
    // Allocate GPU memory
    cudaMalloc((void**) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void**) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    
    // and copy over the relevant data structures to the GPU
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // Create streams for overlapping and define segment size variables
    cudaStream_t stream[num_streams];
    float segsize = ceil(float(Batch) / num_streams);
    int in_segsize = segsize * Channel * Height * Width;
    int out_segsize = segsize * Map_out * Height_out * Width_out;
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // allocate pinned mem for staging and copy data from host buffer
    float* pinned_input; 
    float* pinned_output;
    cudaMallocHost((void **)&pinned_input, Batch * Channel * Height * Width * sizeof(float));
    cudaMallocHost((void **)&pinned_output, Batch * Map_out * Height_out * Width_out * sizeof(float));
    for(int i = 0; i < 10; i++) {
        cudaMemcpyAsync(pinned_input + i * in_segsize, host_input + i * in_segsize, in_segsize * sizeof(float), cudaMemcpyHostToHost, stream[i]);
    }

    // Set the kernel dimensions and call the kernel
    int H_grid = (Height_out + (TILE_WIDTH - 1)) / TILE_WIDTH; // Get height in terms of tiles
    int W_grid = (Width_out + (TILE_WIDTH - 1)) /TILE_WIDTH; // get width in terms of tiles
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    //Griddim changes to have "segsize" number of blocks in x direction (segsize batches per call)
    dim3 gridDim(segsize, Map_out, H_grid * W_grid);

    //overlap kernel calls
    for(int i = 0; i < num_streams; i++) {
        cudaMemcpyAsync(*device_input_ptr + in_segsize * i, pinned_input + in_segsize * i, in_segsize * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>(*device_output_ptr + out_segsize * i, *device_input_ptr + in_segsize * i, *device_mask_ptr, segsize, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync(pinned_output + out_segsize * i, *device_output_ptr + out_segsize * i, out_segsize * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    // Copy data from pinned memory to main host buffer
    for(int i = 0; i < 10; i++) {
        cudaMemcpyAsync((float *)host_output + out_segsize * i, pinned_output + out_segsize * i, out_segsize * sizeof(float), cudaMemcpyHostToHost, stream[i]);
    }

    // destroy streams after use
    for(int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }

    // free all pinned memory
    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
    // free all device memory
    cudaFree(*device_input_ptr);
    cudaFree(*device_mask_ptr);
    cudaFree(*device_output_ptr);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Empty to allow for overlap
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Empty to allow for overlap
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