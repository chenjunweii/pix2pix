#ifndef LOSS_HH
#define LOSS_HH

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "network.hh"
#include "config.h"

using namespace std;
using namespace mxnet::cpp;

namespace loss{
	
	Symbol cross_entropy(Symbol p, Symbol g){
		
		//return (0 - (log(p) * g) - log(1 - p) * (1 - g));	
		return (p - g) * (p - g);
	}


	Symbol L1(Symbol p, Symbol g){

		return abs(p - g);

	}
}

#endif
