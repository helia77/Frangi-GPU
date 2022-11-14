#include<iostream>
#include<string>
#include<cstring>
#include<cstdlib>
#include<ctime>
#include<vector>
#include"frangi.h"
#include "./Header.cuh"

using namespace std;

int main()
{
	//construct Hessian kernels
	float beta = 0, c = 0, step = 0, start = 0, end = 0;
	bool black;
	std::cout << "Enter \nBeta1: "; cin >> beta;
	cout << "c: "; cin >> c;
	cout << "Scale: "; cin >> start;
	cout << "Scale end point: "; cin >> end;
	cout << "Steps: "; cin >> step;
	cout << endl;
	//cout << "Is the background black? " << endl;
		//cin >> black;
	cv::Mat input_img = cv::imread("gr.png", cv::IMREAD_GRAYSCALE);
	cv::Mat input_img_fl;
	input_img.convertTo(input_img_fl, CV_32FC1);

	/////////////////////////////////////////////////////////////////////////

	

	int device{};
	std::cout << "Host(0) or Device(1)?" << std::endl;
	std::cin >> device;
	cv::Mat Dxx, Dyy, Dxy;
	cv::Mat vesselness, scale, angles;

	if(device)
		frangi2d(input_img_fl, vesselness, scale, angles, beta, c, start, end, step, true, true);
	else
		frangi2d(input_img_fl, vesselness, scale, angles, beta, c, start, end, step, true, false);
	

	double min, max;
	cv::minMaxLoc(vesselness, &min, &max);
	std::cout << "Min: " << min * 255 << "\t  Max: " << max * 255 << endl;

	cv::imwrite("out.png", vesselness * 255);

    //double A[3], B[3], C[3];

    //A[0] = 5; A[1] = 8; A[2] = 3;
    //B[0] = 7; B[1] = 6; B[2] = 4;
	// 
    //kernel(A, B, C, 3);
}