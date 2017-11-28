#ifndef NETWORK_H
#define NETWORK_H


#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include "flt.h"

using namespace std;
using namespace mxnet::cpp;

namespace network{

	inline void VGG16_Deprecated(char * prefix,
			char * inputs,
			int nbatch,
			map <string, Symbol> * nodes,
			map <string, Symbol> * weights,
			map <string, Symbol> * bias,
			cv::Size size);

	inline void DEVGG16_Deprecated(char * prefix, 
			char * inputs,
			map <string, Symbol> * nodes,
			map <string, Symbol> * weights,
			map <string, Symbol> * bias,
			cv::Size size);

	inline Symbol VGG16(Symbol * inputs,
			Symbol * condition,
			int nbatch,
			map <string, Symbol> * weights,
			map <string, Symbol> * bias,
			map <string, Symbol> * aux,
			cv::Size size);

	inline vector <Symbol> pix2pix_D(Symbol * inputs,
			Symbol * condition,
			int nbatch,
			map <string, Symbol> * weights,
			map <string, Symbol> * bias,
			map <string, Symbol> * aux,
			cv::Size size);

	inline Symbol DEVGG16(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, map <string, Symbol> * aux, cv::Size size);

	inline Symbol MLP(Symbol * inputs,
			Symbol * condition,
			int nbatch,
			map <string, Symbol> * weights,
			map <string, Symbol> * bias,
			cv::Size size);

	inline Symbol DEMLP(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, cv::Size size);

	inline Symbol UNet(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, map <string, Symbol> * aux, cv::Size size);
}


#endif
