#include "cuda_runtime.h"
#include "./Header.cuh"

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize) {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 

void gpu_conv(char* src, char* dst, int sigma, int size,) {
    std::cout << "--------------------- GPU version ---------------------" << std::endl;

    int k_size = 6 * sigma;  //calculate k
    if (k_size % 2 == 0) k_size++; //make sure k is odd
    float miu = k_size / 2;


    cudaDeviceProp props;																//declare a CUDA properties structure
    HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));									//get the properties of the first CUDA device
    
    float* gpu_gKernel;																				//pointer to the gaussian kernel
    char* gpu_inArray;  																	//pointer to input Array
    char* gpu_outArray_x;  																//pointer to output Array after convolution along x
    char* gpu_outArray_y;  																//pointer to output Array after all the convolution
    
    std::cout << "size: " << size << std::endl;
    
    HANDLE_ERROR(cudaMalloc(&gpu_gKernel, k_size * sizeof(float)));  							//allocate memory on device
    HANDLE_ERROR(cudaMalloc(&gpu_inArray, size * sizeof(char)));  							    //allocate memory on device
    HANDLE_ERROR(cudaMalloc(&gpu_outArray_x, size_x * sizeof(char)));  							//allocate memory on device
    HANDLE_ERROR(cudaMalloc(&gpu_outArray_y, size_y * sizeof(char)));  							//allocate memory on device
    
    HANDLE_ERROR(cudaMemcpy(gpu_gKernel, gKernel, k_size * sizeof(float), cudaMemcpyHostToDevice));  //copy the array from main memory to device
    HANDLE_ERROR(cudaMemcpy(gpu_inArray, imageArray, size * sizeof(char), cudaMemcpyHostToDevice));     //copy the array from main memory to device
    
    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    std::cout << "threads.x: " << threads.x << std::endl;
    std::cout << "threads.y: " << threads.y << std::endl;
    dim3 blocks(width / threads.x + 1, height / threads.y + 1);
    
    
    // without shared memory
    std::cout << "Starts doing convolution on GPU without shared memory" << std::endl;
    
    //	utilize cudaEvent_t to serve as GPU timer
    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, NULL);
    
    kernelConvolution_x_ns << <blocks, threads >> > (gpu_outArray_x, gpu_inArray, gpu_gKernel, height, width, height_x, width_x, k_size);
    kernelConvolution_y_ns << <blocks, threads >> > (gpu_outArray_y, gpu_outArray_x, gpu_gKernel, height_x, width_x, height_y, width_y, k_size);
    
    HANDLE_ERROR(cudaMemcpy(outArray_x, gpu_outArray_x, size_x * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory
    HANDLE_ERROR(cudaMemcpy(outArray_y, gpu_outArray_y, size_y * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory
    
    //	end of cudaEvent_t, calculate the time and show
    cudaEventRecord(d_stop, NULL);
    cudaEventSynchronize(d_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, d_start, d_stop);
    std::cout << "It takes " << elapsedTime << " ms to do the GPU based convolution!" << std::endl;
    
    // output ppm file
    char outFile_gpu_x_ns[50] = "./src/gpu_out_x_ns.ppm";
    write_tga("out_GPU_x.tga", outArray_x, width_x, height_x);
    char outFile_gpu_ns[50] = "./src/gpu_out_ns.ppm";
    write_tga("out_GPU_y.tga", outArray_x, width_x, height_x);
    
    std::cout << "Convolution on GPU without shared memory finished" << std::endl;
}
/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
void kernel(double* A, double* B, double* C, int arraySize) {

    // Initialize device pointers.
    double* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, arraySize * sizeof(double));
    cudaMalloc((void**)&d_B, arraySize * sizeof(double));
    cudaMalloc((void**)&d_C, arraySize * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / arraySize + 1, 1);

    // Launch CUDA kernel.
    vectorAdditionKernel <<<gridSize, blockSize>>> (d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
}