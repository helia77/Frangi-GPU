#include<iostream>
#include"cuda_runtime.h"
#include<cuda_runtime_api.h>
#include"device_launch_parameters.h"
#include"tira/image.h"
#include<tira/image/colormap.h>
//#include"frangi.h"

#define _USE_MATH_DEFINES
#define __CUDACC__
# define PI 3.14159265358979323846  /* pi */

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 


//  convolution on device
__global__ void dev_conv(float* out, float* img, float* kernel, int img_w, int out_h, int out_w, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    
    // i and j being smaller than output's width and height, manage the edges perfectly
    if (i >= out_h || j >= out_w) return;

    float conv = 0;
    for (int ki = 0; ki < K; ki++)
        for (int kj = 0; kj < K; kj++)
            conv += img[(i + ki) * img_w + j + kj] * kernel[ki*K + kj];
        
    out[i * out_w + j] = conv;

}

// convolving a kernel with an image using GPU
tira::image<float> convolution_gpu(tira::image<float>& img, float* kernel, int k_size) {

    tira::image<float> src(img);
    int size = src.width() * src.height();		        // size of the image
    
    // output sizes after convolution
    int y_height = src.height() - k_size + 1;
    int y_width = src.width() - k_size + 1;
    int y_size = y_height * y_width;
    float* y_output = (float*)malloc(y_size * sizeof(float));

    // -------------------------------------- GPU ---------------------------------------- //
    int d;
    HANDLE_ERROR(cudaGetDevice(&d));
    //std::cout << "Current device: " << d << std::endl;
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, d));
    
    float* gpu_gKernel;
    float* gpu_image;
    float* gpu_output_x;
    float* gpu_output_y;
    
    // allocate memory for image, kernel, and convoled output
    HANDLE_ERROR(cudaMalloc(&gpu_gKernel, k_size * k_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpu_image, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));

    float* imageArr = img.data();
    float* gKernel = kernel;

    // copy image and kernel from main memory to Device
    HANDLE_ERROR(cudaMemcpy(gpu_image, imageArr, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_gKernel, gKernel, k_size * k_size * sizeof(float), cudaMemcpyHostToDevice));


    size_t blockDim = sqrt(prop.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    //std::cout << "w: " << width << std::endl << "h: " << height << std::endl;
    dim3 blocks(src.width() / threads.x + 1, src.height() / threads.y + 1);

    // starting GPU timer
    /*cudaEvent_t g_start;
    cudaEvent_t g_stop;
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventRecord(g_start, NULL);*/

    // convolving
    dev_conv << < blocks, threads >> > (gpu_output_y, gpu_image, gpu_gKernel, src.width(), y_height, y_width, k_size);
    
    // copy back the results to main memory
    HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));

    // GPU timer ends
    /*cudaEventRecord(g_stop, NULL);
    cudaEventSynchronize(g_stop);
    float eTime;
    cudaEventElapsedTime(&eTime, g_start, g_stop);
    std::cout << "Takes " << eTime << " ms to convolve on GPU" << std::endl;*/

    tira::image<float> output(y_output, y_width, y_height);
    
    return output;
}