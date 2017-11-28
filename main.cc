#include <map>
#include <vector>
#include <random>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <boost/program_options.hpp>

#include "flt.h"
#include "config.h"
#include "pix2pix.hh"

using namespace std;
using namespace mxnet::cpp;

namespace po = boost::program_options;

int main(int argc, char ** argv){
	
	po::options_description desc("Options");
	
	//desc.add_options() ("help", "Print help messages");
	
	desc.add_options() ("restore,r", po::value <string> () -> default_value(""), "restore checkpoint");

	desc.add_options() ("image, i", po::value <string> () -> default_value(""), "test image");

	desc.add_options() ("gpu,g", po::value <int> () -> default_value(0), "GPU ID");
	
	po::variables_map vm;

	try {
        
		po::store(po::parse_command_line(argc, argv, desc), vm);
        
		po::notify(vm);

    } catch (po::error& e) {
        
		cerr << "ERROR: " << e.what() << endl << endl << desc << endl;
        
		return 1;
    }


	config c;

	c.size.height = 256;
	
	c.size.width = 256;
	
	c.nbatch = 1;
	
	//c.nobject = 10;
	
	//c.nnoise = 400;

	c.sdataset = string("cityscapes");
	
	c.pretrained = ("vgg16_weights.h5");

	c.checkpoint = vm["restore"].as <string> ();

	c.slist = string("train_multi.txt");

	c.debug = true;
	
	c.device = DeviceType::kGPU;

	flt::fdebug::log("before cgan instance ...", c.debug);
	
	pix2pix p(c);

	flt::fdebug::log("after cgan instance ...", c.debug);

	p.build();

	int gpu_id = vm["gpu"].as <int> ();
	
	p.train(1000000, gpu_id);
	
	//p.test(vm["image"].as <string> ());
	
	cout << "Done ... " << endl;

	return 0;
}


