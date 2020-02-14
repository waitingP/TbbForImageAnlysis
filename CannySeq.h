#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

/*
 * This class implements Canny Edge Detection using
 * sequential programming, i.e. traditional for loops
 */
class CannySeq {
public:
	CannySeq();
	virtual ~CannySeq();

public:
	void ApplyEdgeDetection(Mat& inFrame, Mat& outFrame);

private:
	void GaussianFilter(Mat& inFrame, Mat& outFrame);
	void SobelOperator(Mat& inFrame, Mat& outFrame, Mat& angleGrad);
	void GradientTrace(Mat& outFrame, Mat& angleGrad);
	void DoubleThreshold(Mat& outFrame);	
	void Hysteresis(Mat& outFrame);

	//Utility functions
	//copies image edge data as it is
	void CopyEdgePixelData(Mat& inFrame, Mat& outFrame);	
};
