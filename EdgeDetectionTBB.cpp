// EdgeDetectionTBB.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <fstream>
#include "CannySeq.h"
#include "CannyTBB.h"

#include "opencv2/opencv.hpp"
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace cv;
int main(int argc, char** argv)
{	
	if (argc != 5)
	{
		cout << " Usage: EdgeDetectionTbb -mode seq/tbb/opencv -inp <video path>" << endl;
		return -1;
	}

	string modeStr = argv[1];
	string mode = argv[2];
	string videoPathStr = argv[3];
	string videoPath = argv[4];
	
	if( modeStr != "-mode")
	{
		cout << "Usage: EdgeDetectionTbb - mode seq/tbb/opencv -inp <video path>" << endl;
		return -1;
	}
		
	if (mode != "seq" && mode != "tbb" && mode != "opencv")
	{
		cout << "Usage: EdgeDetectionTbb - mode seq/tbb/opencv -inp <video path>" << endl;
		return -1;
	}

	if(videoPathStr != "-inp")
	{
		cout << "Usage: EdgeDetectionTbb - mode seq/tbb/opencv -inp <video path>" << endl;
		return -1;
	}

	
	if (videoPath.length() == 0)
	{
		cout << "Usage: EdgeDetectionTbb - mode seq/tbb/opencv -inp <video path>" << endl;
		return -1;
	}

	//read video
	VideoCapture video(videoPath);

	
	if (!video.isOpened()) {
		cout << "Error opening video file" << endl;
		return -1;
	}

	ofstream timeLog;
	timeLog.open("TimeLog.txt", ios::app);
	if (!timeLog.is_open())
		cout << "file not found";
	//note start time
	auto start = high_resolution_clock::now();

	int numFrames = 0;
	while (1) {

		Mat frame;
		// Capture frame-by-frame
		video >> frame;		
		// If the frame is empty, break immediately
		if (frame.empty())
			break;
			
		numFrames++;
		Mat grayFrm;
		cvtColor(frame, grayFrm, COLOR_BGR2GRAY);

		Mat cannyFrm = Mat::zeros(grayFrm.rows, grayFrm.cols, CV_8UC1);
		if (mode == "opencv")
		{
			Canny(grayFrm, cannyFrm, 60, 60 * 3);
			// Display the edge detected frame			
			imshow("Frame", cannyFrm);
		}
		else
		{			
			if (mode == "seq")
			{
				//apply sequential edge detection
				CannySeq seqApp;
				seqApp.ApplyEdgeDetection(grayFrm, cannyFrm);				

			}
			else if (mode == "tbb")
			{
				CannyTBB tbbApp;
				tbbApp.ApplyEdgeDetectionTBB(grayFrm, cannyFrm);
			}
			
			// Display the edge detected frame
			imshow("Original", grayFrm);
			imshow("Canny", cannyFrm);
		}		

		// Press  ESC on keyboard to exit
		char c = (char)waitKey(1);
		if (c == 27)
			break;
	}

	//note end time
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//log the time in file
	timeLog << "Time log for " << videoPath << endl;
	timeLog << "Total time for " << mode << ": " << duration.count() << " microseconds" << endl;
	long long avgTime = duration.count()/ numFrames;
	timeLog << "Average time per frame : " << avgTime << " microseconds" << endl;
	timeLog << "-------------------" << endl;
	timeLog.close();

	//release the video capture object
	video.release();

	// Closes all the frames
	destroyAllWindows();
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
