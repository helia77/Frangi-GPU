#define _USE_MATH_DEFINES
#include<iostream>
#include"tira/image.h"
#include<tira/image/colormap.h>

# define PI 3.14159265358979323846  /* pi */
const bool DEBUG = false;
const bool USE_GPU = true;
using namespace std;

tira::image<float> convolution_gpu(tira::image<float>& src, float* kernel, int k_size);


//construct Hessian kernels to later convolve with the image
void make_HKernels(float* kern_xx_f, float* kern_xy_f, float* kern_yy_f, float scale) {
	int half_kernel = (int)ceil(3 * scale);
	int n_kern_x = 2 * half_kernel + 1;
	int n_kern_y = n_kern_x;

	int i = 0, j = 0;
	float scale_2 = scale * scale;
	float scale_4 = scale_2 * scale_2;
	float scale_6 = scale_2 * scale_4;
	float half_PI_scale_4 = 1.0f / (2.0f * (float)PI * scale_4);
	float half_PI_scale_6 = 1.0f / (2.0f * (float)PI * scale_6);
	float x2{};
	for (int x = -half_kernel; x <= half_kernel; x++) {
		j = 0;
		x2 = x * x;
		for (int y = -half_kernel; y <= half_kernel; y++) {
			float y2 = y * y;
			kern_xx_f[i * n_kern_y + j] = half_PI_scale_4 * (x2 / scale_2 - 1) * exp(-(x2 + y2) / (2.0f * scale_2));
			kern_xy_f[i * n_kern_y + j] = half_PI_scale_6 * (x * y) * exp(-(x2 + y2) / (2.0f * scale_2));
			j++;
		}
		i++;
	}

	// kern_yy_f is transpose(kern_xx_f)
	for (int z = 0; z < n_kern_y; z++)
		for (int i = 0; i < n_kern_x; i++)
			kern_yy_f[z * n_kern_x + i] = kern_xx_f[i * n_kern_x + z];

}

// Run 2D Hessian filter with parameter sigma on src, return 3 channel image, comprising Dxx, Dxy, Dyy.
tira::image<float> frangi_hessian(tira::image<float> src, float scale) {
	
	int half_kernel = (int)ceil(3 * scale);
	int n_kern_x = 2 * half_kernel + 1;
	int n_kern_y = n_kern_x;
	float* kern_xx_f = new float[n_kern_x * n_kern_y]();				// kernel for convolving with image to create Dxx matrix
	float* kern_xy_f = new float[n_kern_x * n_kern_y]();				// kernel for convolving with image to create Dxy matrix
	float* kern_yy_f = new float[n_kern_x * n_kern_y]();				// kernel for convolving with image to create Dyy matrix

	make_HKernels(kern_xx_f, kern_xy_f, kern_yy_f, scale);				// make Hessian mtx and store in (xx, xy, yy) matrices

	tira::image<float> padded_img(src.border_replicate(half_kernel));		// padding the image for same size results for each scale

	clock_t start, stop;

	//								 -----------
	//								| Dxx | Dxy |
	//		Hessian					|----- -----|
	//								| Dxy | Dyy |
	//								 -----------

	tira::image<float> Dxx, Dxy, Dyy;

	if (USE_GPU) {														// convolving on GPU
		start = clock();
		
		Dxx = convolution_gpu(padded_img, kern_xx_f, n_kern_x);
		Dxy = convolution_gpu(padded_img, kern_xy_f, n_kern_x);
		Dyy = convolution_gpu(padded_img, kern_yy_f, n_kern_x);

		stop = clock();  // time finishes
		std::cout << "Took " << (double)(stop - start) / CLOCKS_PER_SEC << " s to convolve on GPU" << std::endl;
	}
	else {																// convolving on CPU
		start = clock();
		
		// image(T* data, size_t x, size_t y, size_t c = 1)
		Dxx = padded_img.convolve2(tira::image(kern_xx_f, n_kern_x, n_kern_y));
		Dxy = padded_img.convolve2(tira::image(kern_xy_f, n_kern_x, n_kern_y));
		Dyy = padded_img.convolve2(tira::image(kern_yy_f, n_kern_x, n_kern_y));

		stop = clock();  // time finishes
		std::cout << "Took " << (double)(stop - start) / CLOCKS_PER_SEC << " s to convolve on CPU" << std::endl;
	}

	if (DEBUG) {														// check how the Dxx, Dxy, and Dyy looks
		tira::colormap::cpu2image(Dxx.data(), "Dxx.bmp", Dxx.width(), Dxx.height());
		tira::colormap::cpu2image(Dyy.data(), "Dyy.bmp", Dyy.width(), Dyy.height());
		tira::colormap::cpu2image(Dxy.data(), "Dxy.bmp", Dxy.width(), Dxy.height());
	}

	// save Dxx, Dxy, Dyy as channels in one image and output
	tira::image<float> image_3(Dxx.width(), Dxx.height(), 3);
	//std::cout << src.width() << " ?= " << Dxx.width() << ", " << src.height() << " ?= " << Dxx.height() << std::endl;
	image_3.channel(Dxx.data(), 0);
	image_3.channel(Dxy.data(), 1);
	image_3.channel(Dyy.data(), 2);

	delete[] kern_xx_f;
	delete[] kern_xy_f;
	delete[] kern_yy_f;

	return image_3;
}

// Helper functions - multiplying all image pixels by a float - for tira::image
tira::image<float> mult(tira::image<float> img, float n) {

	tira::image<float> out(img.width(), img.height(), img.channels());

	for (int i = 0; i < img.width(); i++)
		for (int j = 0; j < img.height(); j++)
			for (int c = 0; c < img.channels(); c++)
				out(i, j, c) = img(i, j, c) * n;
	return out;
}

// Estimate eigenvalues from Dxx, Dxy, Dyy. Save results to lambda1, lambda2, Ix, Iy.
tira::image<float> frangi_eig(tira::image<float> image_3) {

	// 2 stack of eigenvalues for each pixel
	tira::image<float> output(image_3.width(), image_3.height(), 2);

	for (int x = 0; x < image_3.width(); x++) {
		for (int y = 0; y < image_3.height(); y++) {
			// exctract the hessian matrix equivalent of each pixel
			float dxx = image_3(x, y, 0);
			float dxy = image_3(x, y, 1);
			float dyy = image_3(x, y, 2);

			// calculate eigenvectors
			float tmp = dxx - dyy;
			float tmp2 = sqrt(tmp * tmp + (4 * dxy * dxy));

			// compute eigenvalues
			float mu1 = 0.5 * (dxx + dyy + tmp2);
			float mu2 = 0.5 * (dxx + dyy - tmp2);

			// sort eigenvalues and eigenvectors by absolute value abs(lambda1) < abd(lambda2)
			float lambda1 = (abs(mu1) < abs(mu2)) ? mu1 : mu2;
			float lambda2 = (abs(mu1) < abs(mu2)) ? mu2 : mu1;

			output(x, y, 0) = lambda1;
			output(x, y, 1) = lambda2;
		}
	}

	std::cout << "Eigenvalue decomposition done." << std::endl;
	return output;
}

// Comute the first component of the vesselness equation
tira::image<float> term1(tira::image<float> eigs, float beta) {

	tira::image<float> tmp1(eigs.width(), eigs.height());

	for (int i = 0; i < eigs.width(); i++) {
		for (int j = 0; j < eigs.height(); j++) {
			float lambda1 = eigs(i, j, 0);
			float lambda2 = eigs(i, j, 1);

			float Rb = lambda1 / lambda2;

			tmp1(i, j) = exp(-Rb * Rb / beta);
		}
	}

	return tmp1;
}
// Comute the second component of the vesselness equation
tira::image<float> term2(tira::image<float> eigs, float c) {

	tira::image<float> tmp2(eigs.width(), eigs.height());

	for (int i = 0; i < eigs.width(); i++) {
		for (int j = 0; j < eigs.height(); j++) {
			float lambda1 = eigs(i, j, 0);
			float lambda2 = eigs(i, j, 1);

			float S2 = (lambda1 * lambda1) + (lambda2 * lambda2);

			tmp2(i, j) = exp(-S2 / c);
		}
	}

	return tmp2;
}

tira::image<float> filtered_img(tira::image<float> tmp1, tira::image<float> tmp2) {
	
	tira::image<float> output(tmp1.width(), tmp1.height());
	//cv::Mat Ifiltered = tmp1.mul(cv::Mat::ones(src.rows, src.cols, src.type()) - tmp2);
	for (int i = 0; i < tmp1.width(); i++) {
		for (int j = 0; j < tmp1.height(); j++) {
			output(i, j) = tmp1(i, j) * (1.0f - tmp2(i, j));
		}
	}
	std::cout << "Vesselness function done." << std::endl;
	return output;
}

// check if no lambda2 is equal to 0, if yes, given the smallest value possible
void check_zero(tira::image<float>& eig_2) {

	for (int i = 0; i < eig_2.width(); i++)
		for (int j = 0; j < eig_2.height(); j++)
			if (eig_2(i, j, 1) == 0)				
				eig_2(i, j, 1) = nextafterf(0, 1);

}

void neg_lambda(tira::image<float>& img, tira::image<float> lambda2) {

	for (int i = 0; i < img.width(); i++)
		for (int j = 0; j < img.height(); j++)
			if(lambda2(i, j) > 0)
				img(i, j) = 0;
}

// Apply full Frangi filter to src.Vesselness is saved in J, scale is saved to scale, vessel angle is saved to directions.
tira::image<float> frangi(tira::image<float> src, float B, float C, float start, float end,
	float step) {

	vector<tira::image<float>> ALLfiltered;
	//vector<float> ALLangles;

	float beta = 2 * B * B;			// based on the formula
	float c = 2 * C * C;			// based on the formula
	int cnt = 1;

	for (float scale = start; scale <= end; scale += step) {
		std::cout << "\nScale #" << cnt << std::endl;
		//create 2D hessians
		tira::image<float> D = frangi_hessian(src, scale);
		
		//correct for scale - D * scale * scale
		D = mult(mult(D, scale), scale);

		//calculate (absolute value sorted) eigenvalues and vectors
		tira::image<float> eig_2 = frangi_eig(D);
		
		// Check lambda2 > 0 and if equal to zero, make the smallest floating point value possible
		check_zero(eig_2);
		
		// Create the two terms in the vesselness equation
		tira::image<float> tmp1 = term1(eig_2, beta);
		tira::image<float> tmp2 = term2(eig_2, c);

		// Compute filtered image
		tira::image<float> filtered1 = filtered_img(tmp1, tmp2);

		neg_lambda(filtered1, eig_2.channel(1));

		//store results
		ALLfiltered.push_back(filtered1);
		std::cout << "Filter number " << std::to_string(cnt) << ": \n\t min: " << filtered1.minv() << "\t max: " << filtered1.maxv() << std::endl;
		cnt++;
	}

	tira::image<float> maxVals = ALLfiltered[0];
	float max_v = 0;
	int b = ALLfiltered.size() - 1;

	tira::image<float> output_image(maxVals.width(), maxVals.height());

	//find element-wise maximum across all accumulated filter results
	for (int x = 0; x < maxVals.width(); x++)
		for (int y = 0; y < maxVals.height(); y++) {
			max_v = maxVals(x, y);
			for (int i = 1; i < ALLfiltered.size(); i++)
				if (ALLfiltered[i](x, y) > max_v)
					max_v = ALLfiltered[i](x, y);

			output_image(x, y) = max_v;
		}
	
	//return mult(output_image, 255);
	return output_image;
}


int main() {

    float beta = 1, c = 20, step = 0.5, start = 1, end = 3;

    tira::image<float> loaded_image("brain.ppm");
	tira::image<float> img = loaded_image.channel(0);
	tira::image<float> output_img = frangi(img, beta, c, start, end, step);
	tira::colormap::cpu2image(output_img.data(), "result.bmp", output_img.width(), output_img.height(),
		tira::colormap::cmGrayscale);
}