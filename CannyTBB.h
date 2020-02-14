#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

/*
 * This class implements Canny Edge Detection using 
 * Intel Threading Building Blocks
 */
class CannyTBB
{
public:
	CannyTBB();
	~CannyTBB();

public:
	void ApplyEdgeDetectionTBB(Mat& inFrame, Mat& outFrame);

private:
	void GaussianFilter(Mat& inFrame, Mat& outFrame);
	void SobelOperator(Mat& inFrame, Mat& outFrame, Mat& angleGrad);
	void GradientTrace(Mat& outFrame, Mat& angleGrad);		
	void DoubleThreshold(Mat& outFrame);
	void Hysteresis(Mat& outFrame);
};

