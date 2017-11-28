#ifndef INIT_H
#define INIT_H

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include "pix2pix.h"
#include "flt.h"

using namespace std;
using namespace mxnet::cpp;

namespace bpo = boost::program_options;


namespace init{

	enum init_mode{
		
		restore, pretrained, predict

	};

	
	/* 	
	 *
	 *  		accept "node name vector" and "node shape vector" 
	 *
	 *			paramaters :
	 *
	 *				node => node name vetor
	 *
	 *				shape => node shape vector
	 *				
	 *				filename => pretrained weight file
	 *
	 *				pretrained => specify which node should initialize with pretrained weight;
	 *
	 *	
	 *
	 */

	inline void init_weight_simple(vector <string> &node,
		vector <vector <mx_uint>> &shape,
		map <string, NDArray> &ndarg,
		map <string, NDArray> &grad,
		Context ctx,
		init_mode im,
		string filename = "",
		vector <string> * pretrained = nullptr,
		map <string, string> * mapping = nullptr,
		bool node_mapping = false);

	inline void init_weight(vector <string> &node,
		vector <mx_shape> &shape,
		vector <NDArray> &ndarg,
		vector <NDArray> &grad,
		Context ctx,
		init_mode im,
		string filename = "",
		vector <string> * pretrained = nullptr,
		map <string, string> * mapping = nullptr,
		bool node_mapping = false);

	inline vector <OpReqType> wrt(char * prefix,
		vector <string> &node);

	inline vector <OpReqType> wrt(vector <char *> prefix,
		vector <string> &node);


	inline void init_aux(vector <string> &aux,
		vector <vector <mx_uint>> &shape,
		map <string, NDArray> &ndaux,
		Context ctx,
		init_mode im,
		string filename = "");
}



#endif
