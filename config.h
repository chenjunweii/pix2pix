#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>

using namespace std;
using namespace mxnet::cpp;

struct config{
	
	cv::Size size;
	
	int nbatch;
	int nobject;
	int nnoise;

	string sdataset;

	string slist;
	
	string pretrained;
	
	string checkpoint;

	vector <string> label;

	bool debug;

	DeviceType device;

};



#endif
