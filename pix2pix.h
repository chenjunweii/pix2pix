#ifndef PIX2PIX_H
#define PIX2PIX_H

#include <map>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <boost/program_options.hpp>

#include "network.hh"
#include "config.h"

using namespace std;
using namespace mxnet::cpp;


typedef vector <mx_uint> mx_shape;
typedef vector <vector <mx_uint>> vmx_shape; // input shape



class pix2pix{
	
	public:

		map <string, Symbol> node;
		
		map <string, Symbol> weight;
		
		map <string, Symbol> bias;
		
		map <string, Symbol> aux;
		
		map <string, NDArray> ndarg;
		
		map <string, vector <mx_uint>> inputs_shape;
		
		cv::Size size;

		string slist;

		string sdataset;
		
		string pretrained;
		
		string checkpoint;

		vector <string> label;

		int nbatch; // number of training batch
		
		int nobject; // number of object
		
		int nclass; // number of classes
		
		int nnoise; // noise size
		
		DeviceType device;
		
		Symbol inputs = flt::fmx::fimage::encodeb(Symbol::Variable("inputs"));
		
		Symbol z = Symbol::Variable("z");

		Symbol c = flt::fmx::fimage::encodeb(Symbol::Variable("c"));
		
//		Symbol generated = Symbol::Variable("generated"); // not same as node["generated"]

		//vector <Symbol> condition = vector <Symbol> (2);// = Symbol::Variable("condition");
	
		pix2pix (config);

		~pix2pix();

		inline void build();

		inline void train(int iters, int device_id);

		inline void test(string image);
		
		inline Symbol D_Loss();
		
		inline Symbol G_Loss();
};

#endif
