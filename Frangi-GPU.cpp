#include<iostream>
#include<string>
#include<cstring>
#include<cstdlib>
#include<ctime>
#include<vector>
#include"frangi.h"
#include "frangi.cuh"

using namespace std;

int main()
{
	float beta = 1.5, c = 15, step = 1, start = 5, end = 10;
	bool black;
	/*std::cout << "Enter \nBeta1: "; cin >> beta;
	cout << "c: "; cin >> c;
	cout << "Scale: "; cin >> start;
	cout << "Scale end point: "; cin >> end;
	cout << "Steps: "; cin >> step;
	cout << endl;*/
	//cout << "Is the background black? " << endl;
		//cin >> black;
	cv::Mat input_img = cv::imread("gr.png", cv::IMREAD_GRAYSCALE);
	cv::Mat input_img_fl;
	input_img.convertTo(input_img_fl, CV_32FC1);
	cv::Mat vesselness, scale, angles;

	/////////////////////////////////////////////////////////////////////////
	bool device = true;
	//frangi2d(input_img_fl, vesselness, scale, angles, beta, c, start, end, step, true, device);
	

	double min, max;
	//cv::minMaxLoc(vesselness, &min, &max);
	//std::cout << "Min: " << min * 255 << "\t  Max: " << max * 255 << endl;

	//cv::imwrite("out.png", vesselness * 255);

}