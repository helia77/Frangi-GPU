#include "device_launch_parameters.h"
#include<iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include<opencv2/opencv.hpp>

void kernel(double* A, double* B, double* C, int arraySize);

void convolution_gpu(cv::Mat& Dxx, const cv::Mat& src, cv::Mat& kernel);