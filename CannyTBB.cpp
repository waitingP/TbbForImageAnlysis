#include "pch.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CannyTBB.h"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
using namespace tbb;
using namespace std;

CannyTBB::CannyTBB() {
}

CannyTBB::~CannyTBB() {
}

void CannyTBB::ApplyEdgeDetectionTBB(Mat &inFrame, Mat &outFrame) {
	/*ofstream timeLog;
	timeLog.open("TimeTbb.txt", ios::app);
	if (!timeLog.is_open())
		cout << "file not found";
	//note start time
	auto start = high_resolution_clock::now();*/

	//call Gaussian blur
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
	timeLog << duration.count() << " microseconds"<< endl;
	timeLog << "-------------------" << endl;
	timeLog.close();*/
}

//filter for x-direction
class GaussianTBBX {
	Mat m_inFrame;
	Mat m_outFrame;
public:
	void operator()(const blocked_range<int> &r) const {
		double coeffs[] = { 0.2442, 0.4026, 0.2442 };
		for (size_t row = r.begin(); row != r.end(); ++row) {
			for (int col = 1; col < m_outFrame.cols - 1; col++) {
				m_outFrame.data[row * m_outFrame.cols + col] =
						m_inFrame.data[row * m_inFrame.cols + col - 1]
								* coeffs[0]
								+ m_inFrame.data[row * m_inFrame.cols + col]
										* coeffs[1]
								+ m_inFrame.data[row * m_inFrame.cols + col + 1]
										* coeffs[2];
			}
		}
	}
	GaussianTBBX(Mat inFrame, Mat outFrame) :
			m_inFrame(inFrame), m_outFrame(outFrame) {
	}
};

//filter for Y- direction
class GaussianTBBY {
	Mat m_inFrame;
	Mat m_outFrame;
public:
	void operator()(const blocked_range<int> &r) const {
		double coeffs[] = { 0.2442, 0.4026, 0.2442 };
		for (size_t col = r.begin(); col != r.end(); ++col) {
			for (int row = 1; row < m_inFrame.rows - 1; row++) {
				m_outFrame.data[row * m_outFrame.cols + col] =
						m_outFrame.data[(row - 1) * m_outFrame.cols + col]
								* coeffs[0]
								+ m_outFrame.data[row * m_outFrame.cols + col]
										* coeffs[1]
								+ m_outFrame.data[(row + 1) * m_outFrame.cols
										+ col] * coeffs[2];
			}
		}
	}
	GaussianTBBY(Mat inFrame, Mat outFrame) :
			m_inFrame(inFrame), m_outFrame(outFrame) {
	}
};
void CannyTBB::GaussianFilter(Mat &inFrame, Mat &outFrame) {
	//copy 0th column and last column as it is
	parallel_for(int(0), outFrame.rows - 1,
			[&](int row) {
				outFrame.data[row * outFrame.cols + 0] = inFrame.data[row
						* inFrame.cols + 0];
				outFrame.data[row * outFrame.cols + outFrame.cols - 1] =
						inFrame.data[row * inFrame.cols + inFrame.cols - 1];
			});
	parallel_for(tbb::blocked_range<int>(0, outFrame.rows - 1),
			GaussianTBBX(inFrame, outFrame));

	//copy 0th row and last row as it is
	parallel_for(int(0), outFrame.cols - 1,
			[&](int col) {
				outFrame.data[col] = inFrame.data[col];
				outFrame.data[(outFrame.rows - 1) * outFrame.cols + col] =
						inFrame.data[(inFrame.rows - 1) * inFrame.cols + col];
			});
	parallel_for(tbb::blocked_range<int>(0, inFrame.cols - 1),
			GaussianTBBY(inFrame, outFrame));
}

class SobelTBB {
	Mat m_inFrame;
	Mat m_outFrame;
	Mat m_angleGrad;
public:
	void operator()(const blocked_range<int> &r) const {

		int xGrad, yGrad, sum;
		for (size_t row = r.begin(); row != r.end(); ++row) {
			for (int col = 1; col < m_outFrame.cols - 1; col++) {

				xGrad = m_inFrame.data[(row - 1) * m_inFrame.cols + col - 1]
						+ 2 * m_inFrame.data[(row) * m_inFrame.cols + col - 1]
						+ m_inFrame.data[(row + 1) * m_inFrame.cols + col - 1]
						- m_inFrame.data[(row - 1) * m_inFrame.cols + col + 1]
						- 2 * m_inFrame.data[(row) * m_inFrame.cols + col + 1]
						- m_inFrame.data[(row + 1) * m_inFrame.cols + col + 1];

				yGrad = m_inFrame.data[(row - 1) * m_inFrame.cols + col - 1]
						+ 2 * m_inFrame.data[(row - 1) * m_inFrame.cols + col]
						+ m_inFrame.data[(row - 1) * m_inFrame.cols + col + 1]
						- m_inFrame.data[(row + 1) * m_inFrame.cols + col - 1]
						- 2 * m_inFrame.data[(row + 1) * m_inFrame.cols + col]
						- m_inFrame.data[(row + 1) * m_inFrame.cols + col + 1];

				sum = abs(xGrad) + abs(yGrad);
				sum = sum > 255 ? 255 : sum;
				sum = sum < 0 ? 0 : sum;
				m_outFrame.data[row * m_outFrame.cols + col] = sum;

				int i = row * m_outFrame.cols + col;

				if (xGrad == 0) {
					if (yGrad > 0) {
						m_angleGrad.data[i] = 90;
					}
					if (yGrad < 0) {
						m_angleGrad.data[i] = -90;
					}
				} else if (yGrad == 0) {
					m_angleGrad.data[i] = 0;
				} else {
					m_angleGrad.data[i] = (float) ((atan(yGrad * 1.0 / xGrad)
							* 180) / M_PI);
				}
				// make it 0 ~ 180
				m_angleGrad.data[i] += 90;
			}
		}
	}
	SobelTBB(Mat inFrame, Mat outFrame, Mat angleGrad) :
			m_inFrame(inFrame), m_outFrame(outFrame), m_angleGrad(angleGrad) {
	}
};
void CannyTBB::SobelOperator(Mat &inFrame, Mat &outFrame, Mat &angleGrad) {
	parallel_for(tbb::blocked_range<int>(1, outFrame.rows - 2),
			SobelTBB(inFrame, outFrame, angleGrad));
}

class GradientTraceTBB {
	Mat m_outFrame;
	Mat m_angleGrad;

public:
	void operator()(const blocked_range<int> &r) const {

		for (size_t row = r.begin(); row != r.end(); ++row) {
			for (int col = 1; col < m_outFrame.cols - 1; col++) {
				int i = row * m_outFrame.cols + col;
				float angle = m_angleGrad.data[i];
				int m0 = m_outFrame.data[row * m_outFrame.cols + col];
				
				int m1 = 255;
				int m2 = 255;
				if (angle >= 0 && angle < 22.5) // angle 0
						{
					m1 = m_outFrame.data[row * m_outFrame.cols + col - 1];
					m2 = m_outFrame.data[row * m_outFrame.cols + col + 1];

				} else if (angle >= 22.5 && angle < 67.5) // angle +45
						{
					m1 = m_outFrame.data[(row - 1) * m_outFrame.cols + col + 1];
					m2 = m_outFrame.data[(row + 1) * m_outFrame.cols + col - 1];

				} else if (angle >= 67.5 && angle < 112.5) // angle 90
						{
					m1 = m_outFrame.data[(row + 1) * m_outFrame.cols + col];
					m2 = m_outFrame.data[(row - 1) * m_outFrame.cols + col];

				} else if (angle >= 112.5 && angle < 157.5) // angle 135 / -45
						{
					m1 = m_outFrame.data[(row - 1) * m_outFrame.cols + col - 1];
					m2 = m_outFrame.data[(row + 1) * m_outFrame.cols + col + 1];

				} else if (angle >= 157.5) // angle 0
						{
					m1 = m_outFrame.data[(row + 1) * m_outFrame.cols + col];
					m2 = m_outFrame.data[(row - 1) * m_outFrame.cols + col];
				}

				if (m0 >= m1 && m0 >= m2) {
					//edge pixel
				} else
					m_outFrame.data[i] = 0;
			}
		}
	}
	GradientTraceTBB(Mat &outFrame, Mat &angleGrad) :
			m_outFrame(outFrame), m_angleGrad(angleGrad) {
	}
};
void CannyTBB::GradientTrace(Mat &outFrame, Mat &angleGrad) {
	parallel_for(tbb::blocked_range<int>(1, outFrame.rows - 2),
			GradientTraceTBB(outFrame, angleGrad));
}

class DoubleThresholdTBB {
	float m_lowThreshold;
	float m_highThreshold;
	int weak = 25;
	int strong = 250;
	Mat m_outFrame;

public:
	void operator()(const blocked_range<int> &r) const {
		for (size_t row = r.begin(); row != r.end(); ++row) {
			for (int col = 0; col < m_outFrame.cols; col++) {
				int pixVal = m_outFrame.data[row * m_outFrame.cols + col];
				if (pixVal >= m_highThreshold)
					m_outFrame.data[row * m_outFrame.cols + col] = strong;
				else if (pixVal < m_lowThreshold)
					m_outFrame.data[row * m_outFrame.cols + col] = 0;

				if (pixVal <= m_highThreshold && pixVal >= m_lowThreshold)
					m_outFrame.data[row * m_outFrame.cols + col] = weak;
			}
		}
	}
	DoubleThresholdTBB(float lowThreshold, float highThreshold, Mat &outFrame) :
			m_lowThreshold(lowThreshold), m_highThreshold(highThreshold), m_outFrame(
					outFrame) {
	}
};
void CannyTBB::DoubleThreshold(Mat &outFrame) {
	double lowThresholdRatio = 0.05;
	int highThreshold = threshold(outFrame, outFrame, 0, 255, THRESH_BINARY | THRESH_OTSU);
	int lowThreshold = highThreshold * lowThresholdRatio;

	parallel_for(tbb::blocked_range<int>(0, outFrame.rows - 1),
			DoubleThresholdTBB(lowThreshold, highThreshold, outFrame));
}

class HysteresisTBB {	
	Mat m_outFrame;	
public:
	void operator()(const blocked_range<int> &r) const {
		int weak = 25;
		int strong = 255;
		for (size_t row = r.begin(); row != r.end(); ++row) {
			for (int col = 0; col < m_outFrame.cols; col++) {
				int pixVal = m_outFrame.data[row * m_outFrame.cols + col];
				if (pixVal == weak) {
					if ((m_outFrame.data[(row + 1) * m_outFrame.cols + col - 1] == strong) ||
						(m_outFrame.data[(row + 1) * m_outFrame.cols + col] == strong) ||
						(m_outFrame.data[(row + 1) * m_outFrame.cols + col + 1] == strong) ||
						(m_outFrame.data[row * m_outFrame.cols + col - 1] == strong) ||
						(m_outFrame.data[row * m_outFrame.cols + col + 1] == strong) ||
						(m_outFrame.data[(row - 1) * m_outFrame.cols + col - 1] == strong) ||
						(m_outFrame.data[(row - 1) * m_outFrame.cols + col] == strong) ||
						(m_outFrame.data[(row - 1) * m_outFrame.cols + col + 1] == strong))
						m_outFrame.data[row * m_outFrame.cols + col] = strong;
					else
						m_outFrame.data[row * m_outFrame.cols + col] = 0;
				}
			}
		}
	}
	HysteresisTBB(Mat &outFrame) :
		 m_outFrame(outFrame) {
	}
};

//Convert weak pixel into strong if pixel around it is strong
void CannyTBB::Hysteresis(Mat &outFrame) {
	parallel_for(tbb::blocked_range<int>(0, outFrame.rows - 1),
		HysteresisTBB(outFrame));
}