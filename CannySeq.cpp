#include "pch.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include "CannySeq.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

const float LOW_THRESHOLD = 2.5;
const float HIGH_THRESHOLD = 7.5;
CannySeq::CannySeq() {
}

CannySeq::~CannySeq() {
}

void CannySeq::ApplyEdgeDetection(Mat &inFrame, Mat &outFrame) {
	/*ofstream timeLog;
	timeLog.open("TimeSeq.txt", ios::app);
	if (!timeLog.is_open())
		cout << "file not found";
	//note start time
	auto start = high_resolution_clock::now();*/

	//Apply Gaussian blur
	Mat gaussianFrm = Mat::zeros(inFrame.rows, inFrame.cols, CV_8UC1);
	GaussianFilter(inFrame, gaussianFrm);

	//Apply sobel operator to find edges
	Mat angleGrad = Mat::zeros(inFrame.rows, inFrame.cols, CV_32FC1);	
	SobelOperator(gaussianFrm, outFrame, angleGrad);

	//Apply non- maximum suppression to thin out edges
	GradientTrace(outFrame, angleGrad);

	//Apply double-thesholding to find weak and strong pixels	
	DoubleThreshold(outFrame);

	//Convert weak pixel into strong if pixel around it is strong
	Hysteresis(outFrame);

	//note end time
	/*auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//log the time in file
	timeLog << duration.count() << " microseconds" << endl;
	timeLog << "-------------------" << endl;
	timeLog.close();*/
}

//copies image edge data as it is
void CannySeq::CopyEdgePixelData(Mat &inFrame, Mat &outFrame) {
	//copy 0th column and last column as it is
	for (int row = 0; row < outFrame.rows; row++) {
		outFrame.data[row * outFrame.cols + 0] = inFrame.data[row * inFrame.cols
				+ 0];
		outFrame.data[row * outFrame.cols + outFrame.cols - 1] =
				inFrame.data[row * inFrame.cols + inFrame.cols - 1];
	}

	//copy 0th row and last row as it is
	for (int col = 0; col < outFrame.cols; col++) {
		outFrame.data[col] = inFrame.data[col];
		outFrame.data[(outFrame.rows - 1) * outFrame.cols + col] =
				inFrame.data[(inFrame.rows - 1) * inFrame.cols + col];
	}
}

//apply Gaussian filter to remove noise
void CannySeq::GaussianFilter(Mat &inFrame, Mat &outFrame) {
	CopyEdgePixelData(inFrame, outFrame);
	// coefficients of 1D gaussian kernel with sigma = 1
	//apply x-Kernal from column 1 to column cols - 2; //last but one column

	float coeffs[] = { 0.2442, 0.4026, 0.2442 };
	for (int row = 0; row < outFrame.rows; row++) {
		for (int col = 1; col < outFrame.cols - 1; col++) {
			outFrame.data[row * outFrame.cols + col] = inFrame.data[row
					* inFrame.cols + col - 1] * coeffs[0]
					+ inFrame.data[row * inFrame.cols + col] * coeffs[1]
					+ inFrame.data[row * inFrame.cols + col + 1] * coeffs[2];
		}
	}

	//apply y-Kernal from row 1 to row rows - 2; //last but one row

	for (int col = 0; col < inFrame.cols; col++) {
		for (int row = 1; row < inFrame.rows - 1; row++) {
			outFrame.data[row * outFrame.cols + col] = outFrame.data[(row - 1)
					* outFrame.cols + col] * coeffs[0]
					+ outFrame.data[row * outFrame.cols + col] * coeffs[1]
					+ outFrame.data[(row + 1) * outFrame.cols + col]
							* coeffs[2];
		}
	}

}

//Apply Sobel filter to find x and y gradients
void CannySeq::SobelOperator(Mat &inFrame, Mat &outFrame, Mat& angleGrad) {
	
	CopyEdgePixelData(inFrame, outFrame);
	
	int xGrad, yGrad, sum;
	//calculate x gradient
	for (int row = 1; row < outFrame.rows - 1; row++) {
		for (int col = 1; col < outFrame.cols - 1; col++) {

			xGrad = inFrame.data[(row - 1) * inFrame.cols + col - 1]
					+ 2 * inFrame.data[(row) * inFrame.cols + col - 1]
					+ inFrame.data[(row + 1) * inFrame.cols + col - 1]
					- inFrame.data[(row - 1) * inFrame.cols + col + 1]
					- 2 * inFrame.data[(row) * inFrame.cols + col + 1]
					- inFrame.data[(row + 1) * inFrame.cols + col + 1];

			yGrad = inFrame.data[(row - 1) * inFrame.cols + col - 1]
					+ 2 * inFrame.data[(row - 1) * inFrame.cols + col]
					+ inFrame.data[(row - 1) * inFrame.cols + col + 1]
					- inFrame.data[(row + 1) * inFrame.cols + col - 1]
					- 2 * inFrame.data[(row + 1) * inFrame.cols + col]
					- inFrame.data[(row + 1) * inFrame.cols + col + 1];

			sum = abs(xGrad) + abs(yGrad);
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			outFrame.data[row * outFrame.cols + col] = sum;

			int i = row * outFrame.cols + col;

			if (xGrad == 0) {
				if (yGrad > 0) {
					angleGrad.data[i] = 90;
				}
				if (yGrad < 0) {
					angleGrad.data[i] = -90;
				}
			} else if (yGrad == 0) {
				angleGrad.data[i] = 0;
			} else {
				angleGrad.data[i] = (float) ((atan(yGrad * 1.0 / xGrad) * 180)
						/ M_PI);
			}
			
			// make it 0 ~ 180
			if(angleGrad.data[i] < 0)
				angleGrad.data[i] += 180;			
		}
	}
}

//Apply non-maximum suppression to thin out the edges
void CannySeq::GradientTrace(Mat &outFrame, Mat& angleGrad) {
	for (int row = 1; row < outFrame.rows - 1; row++) {
		for (int col = 1; col < outFrame.cols - 1; col++) {
			int i = row * outFrame.cols + col;
			float angle = angleGrad.data[i];
			int m0 = outFrame.data[row * outFrame.cols + col];
			
			int m1 = 255;
			int m2 = 255;
			if (angle >= 0 && angle < 22.5) // angle 0
			{
				m1 = outFrame.data[row * outFrame.cols + col - 1];
				m2 = outFrame.data[row * outFrame.cols + col + 1];
			}
			else if (angle >= 22.5 && angle < 67.5) // angle +45
			{
				m1 = outFrame.data[(row - 1) * outFrame.cols + col + 1];
				m2 = outFrame.data[(row + 1) * outFrame.cols + col - 1];
			}
			else if (angle >= 67.5 && angle < 112.5) // angle 90
			{
				m1 = outFrame.data[(row + 1) * outFrame.cols + col];
				m2 = outFrame.data[(row - 1) * outFrame.cols + col];
			}
			else if (angle >= 112.5 && angle < 157.5) // angle 135 / -45
			{
				m1 = outFrame.data[(row - 1) * outFrame.cols + col - 1];
				m2 = outFrame.data[(row + 1) * outFrame.cols + col + 1];
			}
			else if (angle >= 157.5) // angle 0
			{
				m1 = outFrame.data[(row + 1) * outFrame.cols + col];
				m2 = outFrame.data[(row - 1) * outFrame.cols + col];
			}

			if (m0 >= m1 && m0 >= m2)
			{
				//edge pixel
			}
			else
				outFrame.data[i] = 0;
		}
	}
}

//Convert weak pixel into strong if pixel around it is strong
void CannySeq::Hysteresis(Mat &outFrame) {
	int weak = 25;
	int strong = 255;
	for (int row = 0; row < outFrame.rows; row++) {
		for (int col = 0; col < outFrame.cols; col++) {
			int pixVal = outFrame.data[row * outFrame.cols + col];
			if (pixVal == weak) {
				if ((outFrame.data[(row + 1) * outFrame.cols + col - 1] == strong) ||
					(outFrame.data[(row + 1) * outFrame.cols + col] == strong) ||
					(outFrame.data[(row + 1) * outFrame.cols + col + 1] == strong) ||
					(outFrame.data[row * outFrame.cols + col - 1] == strong) ||
					(outFrame.data[row * outFrame.cols + col + 1] == strong) ||
					(outFrame.data[(row - 1) * outFrame.cols + col - 1] == strong) ||
					(outFrame.data[(row - 1) * outFrame.cols + col] == strong) ||
					(outFrame.data[(row - 1) * outFrame.cols + col + 1] == strong))
					outFrame.data[row * outFrame.cols + col] = strong;
				else
					outFrame.data[row * outFrame.cols + col] = 0;
			}
		}
	}
}

void CannySeq::DoubleThreshold(Mat& outFrame)
{
	double lowThresholdRatio = 0.05;

	int weak = 25;
	int strong = 255;

	int highThreshold = threshold(outFrame, outFrame, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	int lowThreshold = highThreshold * lowThresholdRatio;
	for (int row = 0; row < outFrame.rows; row++) {
		for (int col = 0; col < outFrame.cols; col++) {
			int pixVal = outFrame.data[row * outFrame.cols + col];
			if (pixVal >= highThreshold)
				outFrame.data[row * outFrame.cols + col] = strong;
			else if (pixVal < lowThreshold)
				outFrame.data[row * outFrame.cols + col] = 0;
			else			
				outFrame.data[row * outFrame.cols + col] = weak;
		}
	}
}