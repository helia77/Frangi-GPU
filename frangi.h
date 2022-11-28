#pragma once

#include<opencv2/opencv.hpp>
#include "./frangi.cuh"

//Parameters for the filter
	//vessel scales
	//Beta: suppression of blob-like structures. 
	//C: background suppression.
	//Black: enhance black structures if true, otherwise enhance white structures



//Main function - apply full Frangi filter to src. Vesselness is saved in J, scale is saved to scale, vessel angle is saved to directions. 
void frangi2d(const cv::Mat& src, cv::Mat& J, cv::Mat& scale, cv::Mat& directions, float beta, float c, float start, float end,
	float step, bool Black, bool device);

//Helper function - run 2D Hessian filter with parameter sigma on src, save to Dxx, Dxy and Dyy. 
void frangi2d_hessian(const cv::Mat& src, cv::Mat& Dxx, cv::Mat& Dxy, cv::Mat& Dyy, float sigma, bool device);

//Helper function - estimate eigenvalues from Dxx, Dxy, Dyy. Save results to lambda1, lambda2, Ix, Iy. 
void frangi2_eig2image(const cv::Mat& Dxx, const cv::Mat& Dxy, const cv::Mat& Dyy, cv::Mat& lambda1, cv::Mat& lambda2, cv::Mat& Ix, cv::Mat& Iy);