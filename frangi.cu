#include "./frangi.cuh"


# define PI           3.14159265358979323846  /* pi */

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 

// ------------------------------------------------------------------------------------------- //

// convolution on device (without shared memory) along x axis
__global__ void dev_conv_x(unsigned char* out, unsigned char* img, float* kernel, int img_w, int out_h, int out_w, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;	            // row index
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;	            // column index

    // i and j being smaller than output's width and height, manage the edges perfectly
    if (i >= out_h || j >= out_w) return;

    // initialize conv register to store the results after convolution
    float conv = 0.0f;

    // apply the convolution with Gaussian kernel along x
    for (int k = 0; k < K; k++) {
        conv += (unsigned char)img[i * img_w + j + k] * kernel[k];
    }
    out[i * out_w + j] = conv;
}

//  convolution on device (without shared memory) along y axis (same algorithm as dev_conv_x)
__global__ void dev_conv_y(unsigned char* out, unsigned char* img, float* kernel, int img_w, int out_h, int out_w, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    // i and j being smaller than output's width and height, manage the edges perfectly
    if (i >= out_h || j >= out_w) return;
    float conv = 0.0f;
    for (int k{}; k < K; k++)
        conv += (unsigned char)img[(i + k) * img_w + j] * kernel[k];
  
    out[i * out_w + j] = conv;
}


void convolution_gpu(cv::Mat& Dxx, const cv::Mat& src, cv::Mat& kernel) {

    int width = src.cols;
    int height = src.rows;
    int size = width * height;		// size of the image
    int k_size = kernel.rows * kernel.cols;

    unsigned char* imageArray = src.isContinuous() ? src.data : src.clone().data;

    // output sizes after convolution along x
    int x_height = height;
    int x_width = width - kernel.cols + 1;
    int x_size = x_height * x_width;
    unsigned char* x_output = (unsigned char*)malloc(x_size * sizeof(char));

    // output sizes after convolution along y
    int y_height = x_height - kernel.rows + 1;
    int y_width = x_width;
    int y_size = y_height * y_width;
    unsigned char* y_output = (unsigned char*)malloc(y_size * sizeof(char));

    unsigned char* gKernel = kernel.isContinuous() ? kernel.data : kernel.clone().data;
    // -------------------------------------- GPU (without shared memory) ---------------------------------------- //

    std::cout << "\n------------------------- GPU (without shared memory) -------------------------" << std::endl;
    int d;
    HANDLE_ERROR(cudaGetDevice(&d));
    std::cout << "Current device: " << d << std::endl;
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, d));

    float* gpu_gKernel;
    unsigned char* gpu_image;
    unsigned char* gpu_output_x;
    unsigned char* gpu_output_y;

    // allocate memory for image, kernel, and two convoled outputs
    HANDLE_ERROR(cudaMalloc(&gpu_gKernel, k_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpu_image, size * sizeof(char)));
    HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(char)));
    HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(char)));

    // copy image and kernel from main memory to Device
    HANDLE_ERROR(cudaMemcpy(gpu_image, imageArray, size * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_gKernel, gKernel, k_size * sizeof(float), cudaMemcpyHostToDevice));

    size_t blockDim = sqrt(prop.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    dim3 blocks(width / threads.x + 1, height / threads.y + 1);


    // starting GPU timer
    cudaEvent_t g_start;
    cudaEvent_t g_stop;
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventRecord(g_start, NULL);


    std::cout << "Convolving on DEVICE without shared memory..." << std::endl;
    // convolving along x
    dev_conv_x << < blocks, threads >> > (gpu_output_x, gpu_image, gpu_gKernel, width, x_height, x_width, k_size);
    // convolving along y
    dev_conv_y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gpu_gKernel, x_width, y_height, y_width, k_size);

    // copy back the results to main memory
    HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(char), cudaMemcpyDeviceToHost));

    // GPU timer ends
    cudaEventRecord(g_stop, NULL);
    cudaEventSynchronize(g_stop);
    float eTime;
    cudaEventElapsedTime(&eTime, g_start, g_stop);
    std::cout << "Takes " << eTime << " ms to convolve on GPU (without shared memory)" << std::endl;

    // write the output file as targa
    //write_tga("GPU_Output_NoSharedM.tga", y_output, y_width, y_height);
    cv::Mat output = cv::Mat(y_height, y_width, CV_32FC1, y_output);
    Dxx = output;
    std::cout << "Convolution on GPU without shared memory finished." << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    /*-----------------------------------------------------------------------------------------------------------*/



}