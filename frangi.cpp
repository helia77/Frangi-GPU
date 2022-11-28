#define _USE_MATH_DEFINES
#include"frangi.h"
#include<iostream>
#include<cmath>
#include<fstream>
#include<numbers>

using namespace std;


void frangi2d_hessian(const cv::Mat& src, cv::Mat& Dxx, cv::Mat& Dxy, cv::Mat& Dyy, float scale, bool device) {
	//construct Hessian kernels
	int half_kernel = (int)ceil(3 * scale);
	int n_kern_x = 2 * half_kernel + 1;
	int n_kern_y = n_kern_x;
	float* kern_xx_f = new float[n_kern_x * n_kern_y]();
	float* kern_xy_f = new float[n_kern_x * n_kern_y]();
	float* kern_yy_f = new float[n_kern_x * n_kern_y]();
	int i = 0, j = 0;
	float scale_2 = scale * scale;
	float scale_4 = scale_2 * scale_2;
	float scale_6 = scale_2 * scale_4;
	float PI_2 = (float)M_PI * 2.0f;
	float half_PI_scale_4 = 1.0f / (2.0f * PI_2 * scale_4);
	float half_PI_scale_6 = 1.0f / (2.0f * PI_2 * scale_6);
	float exp_x2{};
	float x2{};
	for (int x = -half_kernel; x <= half_kernel; x++) {
		j = 0;
		x2 = x * x;
		for (int y = -half_kernel; y <= half_kernel; y++) {
			kern_xx_f[i * n_kern_y + j] = half_PI_scale_4 * (x2 / scale_2 - 1) * exp(-(x2 + y * y) / (2.0f * scale_2));
			kern_xy_f[i * n_kern_y + j] = half_PI_scale_6 * (x * y) * exp(-(x2 + y * y) / (2.0f * scale_2));
			j++;
		}
		i++;
	}

	// kern_yy_f is transpose(kern_xx_f)
	for (int z = 0; z < n_kern_y; z++) {
		for (int i = 0; i < n_kern_x; i++) {
			kern_yy_f[z * n_kern_x + i] = kern_xx_f[i * n_kern_x + z];
		}
	}

	// flip kernels since kernels aren't symmetric and opencv's filter2D operation performs a correlation, not a convolution
	cv::Mat kern_xx;
	cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xx_f), kern_xx, -1);

	cv::Mat kern_xy;
	cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xy_f), kern_xy, -1);

	cv::Mat kern_yy;
	cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_yy_f), kern_yy, -1);

	//specify anchor since we are to perform a convolution, not a correlation
	cv::Point anchor(n_kern_x - n_kern_x / 2 - 1, n_kern_y - n_kern_y / 2 - 1);

	if (!device) {
		cv::filter2D(src, Dxx, -1, kern_xx, anchor);
		cv::filter2D(src, Dxy, -1, kern_xy, anchor);
		cv::filter2D(src, Dyy, -1, kern_yy, anchor);
	}
	else {
		convolution_gpu(Dxx, src, kern_xx);
		convolution_gpu(Dxy, src, kern_xy);
		convolution_gpu(Dyy, src, kern_yy);
	}
	delete[] kern_xx_f;
	delete[] kern_xy_f;
	delete[] kern_yy_f;
}

void frangi2_eig2image(const cv::Mat& Dxx, const cv::Mat& Dxy, const cv::Mat& Dyy, cv::Mat& lambda1, cv::Mat& lambda2, cv::Mat& Ix, cv::Mat& Iy) {
	//calculate eigenvectors of J, v1 and v2
	cv::Mat tmp, tmp2;
	tmp2 = Dxx - Dyy;
	cv::sqrt(tmp2.mul(tmp2) + 4 * Dxy.mul(Dxy), tmp);
	cv::Mat v2x = 2 * Dxy;
	cv::Mat v2y = Dyy - Dxx + tmp;

	//normalize
	cv::Mat mag;
	cv::sqrt((v2x.mul(v2x) + v2y.mul(v2y)), mag);
	cv::Mat v2xtmp = v2x.mul(1.0f / mag);
	v2xtmp.copyTo(v2x, mag != 0);
	cv::Mat v2ytmp = v2y.mul(1.0f / mag);
	v2ytmp.copyTo(v2y, mag != 0);

	//eigenvectors are orthogonal
	cv::Mat v1x, v1y;
	v2y.copyTo(v1x);
	v1x = -1 * v1x;
	v2x.copyTo(v1y);

	//compute eigenvalues
	cv::Mat mu1 = 0.5 * (Dxx + Dyy + tmp);
	cv::Mat mu2 = 0.5 * (Dxx + Dyy - tmp);

	//sort eigenvalues by absolute value abs(Lambda1) < abs(Lamda2)
	cv::Mat check = abs(mu1) > abs(mu2);
	mu1.copyTo(lambda1); mu2.copyTo(lambda1, check);
	mu2.copyTo(lambda2); mu1.copyTo(lambda2, check);

	v1x.copyTo(Ix); v2x.copyTo(Ix, check);
	v1y.copyTo(Iy); v2y.copyTo(Iy, check);

}


void frangi2d(const cv::Mat& src, cv::Mat& maxVals, cv::Mat& whatScale, cv::Mat& outAngles, float beta, float c, float start, float end,
	float step, bool black, bool device) {
	vector<cv::Mat> ALLfiltered;
	vector<cv::Mat> ALLangles;

	beta = 2 * beta * beta;
	c = 2 * c * c;

	for (float scale = start; scale <= end; scale += step) {
		//create 2D hessians
		cv::Mat Dxx, Dyy, Dxy;
		frangi2d_hessian(src, Dxx, Dxy, Dyy, scale, device);
		//correct for scale
		Dxx = Dxx * scale * scale;
		Dyy = Dyy * scale * scale;
		Dxy = Dxy * scale * scale;

		//calculate (abs sorted) eigenvalues and vectors
		cv::Mat lambda1, lambda2, vx, vy;
		frangi2_eig2image(Dxx, Dxy, Dyy, lambda1, lambda2, vx, vy);

		//compute direction of the minor eigenvector
		cv::Mat angles;
		cv::phase(vx, vy, angles);				//calculates the rotation angle of 2D vector that is formed from the corresponding elements
		ALLangles.push_back(angles);

		//compute some similarity measures
		lambda2.setTo(nextafterf(0, 1), lambda2 == 0);
		cv::Mat Rb = lambda1.mul(1.0 / lambda2);
		Rb = Rb.mul(Rb);
		cv::Mat S2 = lambda1.mul(lambda1) + lambda2.mul(lambda2);

		//compute output image
		cv::Mat tmp1, tmp2;
		cv::exp(-Rb / beta, tmp1);
		cv::exp(-S2 / c, tmp2);

		cv::Mat Ifiltered = tmp1.mul(cv::Mat::ones(src.rows, src.cols, src.type()) - tmp2);

		if (black)
			Ifiltered.setTo(0, lambda2 > 0);
		else
			Ifiltered.setTo(0, lambda2 < 0);

		//store results
		ALLfiltered.push_back(Ifiltered);
	}

	float scale = start;
	ALLfiltered[0].copyTo(maxVals);
	ALLfiltered[0].copyTo(whatScale);
	ALLfiltered[0].copyTo(outAngles);
	whatScale.setTo(scale);

	//find element-wise maximum across all accumulated filter results
	for (int i = 1; i < ALLfiltered.size(); i++) {
		maxVals = max(maxVals, ALLfiltered[i]);
		whatScale.setTo(scale, ALLfiltered[i] == maxVals);
		ALLangles[i].copyTo(outAngles, ALLfiltered[i] == maxVals);
		scale += step;
	}
}