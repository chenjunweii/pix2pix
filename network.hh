#ifndef NETWORK_CC
#define NETWORK_CC

#include <iostream>
#include <vector>
#include <string>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include "flt.h"
#include "network.h"

using namespace std;
using namespace mxnet::cpp;
using namespace flt::fmx;

inline void network::VGG16_Deprecated(char * p, char * inputs, int nbatch, map <string, Symbol> *nodes, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){
	
	layer::conv(p, "conv1_1", inputs, nodes, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::conv(p, "conv1_2", "conv1_1", nodes, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool1", "conv1_2", nodes);
	
	layer::conv(p, "conv2_1", "pool1", nodes, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::conv(p, "conv2_2", "conv2_1", nodes, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool2", "conv2_2", nodes);
	

	layer::conv(p, "conv3_1", "pool2", nodes, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv3_2", "conv3_1", nodes, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv3_3", "conv3_2", nodes, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool3", "conv3_3", nodes, Shape(2,2), Shape(2,2), Shape(1,1));
	
	layer::conv(p, "conv4_1", "pool3", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv4_2", "conv4_1", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv4_3", "conv4_2", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool4", "conv4_3", nodes, Shape(3,3), Shape(1,1), Shape(0,0));

	layer::conv(p, "conv5_1", "pool4", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv5_2", "conv4_1", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv5_3", "conv4_2", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool5", "conv5_3", nodes, Shape(3,3), Shape(2,2), Shape(0,0));
	
	//(*nodes)["pool5_reshape"] = Reshape((*nodes)["pool5"], Shape(nbatch, -1));
	cout << "hhas" << endl;	
	vector <Symbol> condition_vector_d = {Reshape((*nodes)[string(p) + "pool5"], Shape(nbatch, -1)), (*nodes)["fcondition"]};

	layer::concat(p, "condition_concat_d", &condition_vector_d, nodes, 1);
	cout << "aawd" << endl;
	layer::fullyconnected(p, "fc1", "condition_concat_d", nodes, weight, bias, 4096);
	
	(*nodes)[string(p) + "fc1_sigmoid"] = sigmoid((*nodes)[string(p) + "fc1"]);

	layer::fullyconnected(p, "fc2", "fc1", nodes, weight, bias, 4096);

	(*nodes)[string(p) + "fc2_sigmoid"] = sigmoid((*nodes)[string(p) + "fc2"]);
	
	layer::fullyconnected(p, "decision", "fc2_sigmoid", nodes, weight, bias, 1);

	(*nodes)[string(p) + "decision_sigmoid"] = sigmoid((*nodes)[string(p) + "decision"]);
}





inline Symbol network::UNet(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, map <string, Symbol> * aux, cv::Size size){

	bool isTraining = true;
	
	auto deconv_en_1 = layer::conv("deconv_en_1_custom", (*inputs), weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	
	//auto pool1 = layer::maxpool("pool1", deconv_en_1, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_2 = layer::conv("deconv_en_2_custom", deconv_en_1, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(2,2));

	//auto pool2 = layer::maxpool("pool2", deconv_en_2, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_3 = layer::conv("deconv_en_3_custom", deconv_en_2, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(2,2));

	//auto pool3 = layer::maxpool("pool3", deconv_en_3, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_4 = layer::conv("deconv_en_4_custom", deconv_en_3, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2));

	//auto pool4 = layer::maxpool("pool4", deconv_en_4, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_5 = layer::conv("deconv_en_5_custom", deconv_en_4, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2));
	
	//auto pool5 = layer::maxpool("pool5", deconv_en_5, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_6 = layer::conv("deconv_en_6_custom", deconv_en_5, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2));

	//auto pool6 = layer::maxpool("pool6", deconv_en_6, Shape(2,2), Shape(2,2), Shape(0,0));
	
	auto deconv_en_7 = layer::conv("deconv_en_7_custom", deconv_en_6, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1));

	auto pool7 = layer::maxpool("pool7", deconv_en_7, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_8 = layer::conv("deconv_en_8_custom", pool7, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1));

	auto pool8 = layer::maxpool("pool8", deconv_en_8, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_de_1 = layer::deconv("deconv_de_1_custom", pool8, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_1_concated = layer::concat(string("de_1_concated"), vector <Symbol> {deconv_de_1, pool7}, 1);
	
	auto deconv_de_2 = layer::deconv("deconv_de_2_custom", deconv_de_1_concated, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_2_concated = layer::concat(string("de_2_concated"), vector <Symbol> {deconv_de_2, deconv_en_6}, 1);
	
	auto deconv_de_3 = layer::deconv("deconv_de_3_custom", deconv_de_2_concated, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_3_concated = layer::concat(string("de_3_concated"), vector <Symbol> {deconv_de_3, deconv_en_5}, 1);

	auto deconv_de_4 = layer::deconv("deconv_de_4_custom", deconv_de_3_concated, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));

	auto deconv_de_4_concated = layer::concat(string("de_4_concated"), vector <Symbol> {deconv_de_4, deconv_en_4}, 1);
	
	auto deconv_de_5 = layer::deconv("deconv_de_5_custom", deconv_de_4_concated, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));

	auto deconv_de_5_concated = layer::concat(string("de_5_concated"), vector <Symbol> {deconv_de_5, deconv_en_3}, 1);

	auto deconv_de_6 = layer::deconv("deconv_de_6_custom", deconv_de_5_concated, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_6_concated = layer::concat(string("de_6_concated"), vector <Symbol> {deconv_de_6, deconv_en_2}, 1);


	auto deconv_de_7 = layer::deconv("deconv_de_7_custom", deconv_de_6_concated, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_7_concated = layer::concat(string("de_7_concated"), vector <Symbol> {deconv_de_7, deconv_en_1}, 1);


	auto deconv_de_8 = layer::deconv("deconv_de_8_custom", deconv_de_7_concated, weight, bias, aux, 3, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1), Shape(1,1), false);


	return sigmoid(deconv_de_8);

}

inline vector <Symbol> network::pix2pix_D(Symbol *inputs, Symbol *condition, int nbatch, map <string, Symbol> *weight, map <string, Symbol> *bias, map <string, Symbol> *aux, cv::Size size){
	
	bool isTraining = true;

	vector <Symbol> features_vector {(*inputs), (*condition)};

	auto features = layer::concat("features", &features_vector,  1);
	/*	
	auto conv1_1 = layer::conv("conv1_1_custom", features, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv1_2 = layer::conv("conv1_2", conv1_1, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool1 = layer::maxpool("pool1", conv1_2);
	
	auto conv2_1 = layer::conv("conv2_1", pool1, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv2_2 = layer::conv("conv2_2", conv2_1, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool2 = layer::maxpool("pool2", conv2_2);

	auto conv3_1 = layer::conv("conv3_1", pool2, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_2 = layer::conv("conv3_2", conv3_1, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_3 = layer::conv("conv3_3", conv3_2, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool3 = layer::maxpool("pool3", conv3_3, Shape(2,2), Shape(2,2), Shape(1,1));
	
	auto conv4_1 = layer::conv("conv4_1", pool3, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_2 = layer::conv("conv4_2", conv4_1, weight, bias, 512,  Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_3 = layer::conv("conv4_3", conv4_2, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool4 = layer::maxpool("pool4", conv4_3, Shape(3,3), Shape(1,1), Shape(0,0));

	auto conv5_1 = layer::conv("conv5_1", pool4, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_2 = layer::conv("conv5_2", conv5_1,  weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_3 = layer::conv("conv5_3", conv5_2, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool5 = layer::maxpool("pool5", conv5_3, Shape(3,3), Shape(2,2), Shape(0,0));
	

	auto fc1 = layer::fullyconnected("fcsggxxwwfs11", pool5, weight, bias, 10);	
	auto fc1_sigmoid = LeakyReLU(fc1) / 10;

	auto fc2 = layer::fullyconnected("fcss22", fc1_sigmoid, weight, bias, 10);
	auto fc2_sigmoid = LeakyReLU(fc2) / 10;
	
	auto fc3 = layer::fullyconnected("fcss33", fc2_sigmoid, weight, bias, 1);
	auto fc3_sigmoid = sigmoid(fc3);
	*/

	auto conv1_1 = layer::conv("conv1_1_custom", features, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto conv1_2 = layer::conv("conv1_2", conv1_1, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto pool1 = layer::maxpool("pool1", conv1_1);
	
	auto conv2_1 = layer::conv("conv2_1", conv1_1, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto conv2_2 = layer::conv("conv2_2", conv2_1, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto pool2 = layer::maxpool("pool2", conv2_1);

	auto conv3_1 = layer::conv("conv3_1", conv2_1, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto conv3_2 = layer::conv("conv3_2", conv3_1, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	//auto conv3_3 = layer::conv("conv3_3", conv3_2, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto pool3 = layer::maxpool("pool3", conv3_1, Shape(2,2), Shape(2,2), Shape(1,1));
	
	auto conv4_1 = layer::conv("conv4_1", conv3_1, weight, bias, aux, 512, "bn", isTraining, Shape(1,1), Shape(1,1), Shape(1,1));
	auto conv4_2 = layer::conv("conv4_2", conv4_1, weight, bias, 1, Shape(1,1), Shape(1,1), Shape(1,1), false);
	//auto conv4_3 = layer::conv("conv4_3", conv4_2, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	//auto pool4 = layer::maxpool("pool4", conv4_3, Shape(3,3), Shape(1,1), Shape(0,0));

	//auto conv5_1 = layer::conv("conv5_1", pool4, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	//auto conv5_2 = layer::conv("conv5_2", conv5_1,  weight, bias, aux, 512, "bn", isTraining, Shape(1,1), Shape(1,1), Shape(1,1));
	//auto conv5_3 = layer::conv("conv5_3", conv5_2, weight, bias, aux, 1, "bn", isTraining, Shape(1,1), Shape(1,1), Shape(1,1), false);

	//auto pool5 = layer::maxpool("pool5", conv5_3, Shape(3,3), Shape(2,2), Shape(0,0));
	
	//auto fc1 = layer::fullyconnected("fcsggxxwwfs11", conv5_3, weight, bias, 1024);
	
	//auto fc1_sigmoid = sigmoid(fc1);

	//auto fc2 = layer::fullyconnected("fcss22", fc1_sigmoid, weight, bias, 1);

	//auto fc2_sigmoid = sigmoid(fc2);
	
	vector <Symbol> l(1);

	l[0] = sigmoid(conv4_2);

	//l[1] = fc2_sigmoid;

	return l;


}
inline Symbol network::VGG16(Symbol *inputs, Symbol *condition, int nbatch, map <string, Symbol> *weight, map <string, Symbol> *bias, map <string, Symbol> *aux, cv::Size size){
	
	bool isTraining = true;

	vector <Symbol> features_vector {(*inputs), (*condition)};

	auto features = layer::concat("features", &features_vector,  1);
	/*	
	auto conv1_1 = layer::conv("conv1_1_custom", features, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv1_2 = layer::conv("conv1_2", conv1_1, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool1 = layer::maxpool("pool1", conv1_2);
	
	auto conv2_1 = layer::conv("conv2_1", pool1, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv2_2 = layer::conv("conv2_2", conv2_1, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool2 = layer::maxpool("pool2", conv2_2);

	auto conv3_1 = layer::conv("conv3_1", pool2, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_2 = layer::conv("conv3_2", conv3_1, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_3 = layer::conv("conv3_3", conv3_2, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool3 = layer::maxpool("pool3", conv3_3, Shape(2,2), Shape(2,2), Shape(1,1));
	
	auto conv4_1 = layer::conv("conv4_1", pool3, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_2 = layer::conv("conv4_2", conv4_1, weight, bias, 512,  Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_3 = layer::conv("conv4_3", conv4_2, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool4 = layer::maxpool("pool4", conv4_3, Shape(3,3), Shape(1,1), Shape(0,0));

	auto conv5_1 = layer::conv("conv5_1", pool4, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_2 = layer::conv("conv5_2", conv5_1,  weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_3 = layer::conv("conv5_3", conv5_2, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool5 = layer::maxpool("pool5", conv5_3, Shape(3,3), Shape(2,2), Shape(0,0));
	

	auto fc1 = layer::fullyconnected("fcsggxxwwfs11", pool5, weight, bias, 10);	
	auto fc1_sigmoid = LeakyReLU(fc1) / 10;

	auto fc2 = layer::fullyconnected("fcss22", fc1_sigmoid, weight, bias, 10);
	auto fc2_sigmoid = LeakyReLU(fc2) / 10;
	
	auto fc3 = layer::fullyconnected("fcss33", fc2_sigmoid, weight, bias, 1);
	auto fc3_sigmoid = sigmoid(fc3);
	*/

	auto conv1_1 = layer::conv("conv1_1_custom", features, weight, bias, aux, 64, "bn", isTraining, Shape(4,4), Shape(1,1), Shape(1,1));
	auto conv1_2 = layer::conv("conv1_2", conv1_1, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto pool1 = layer::maxpool("pool1", conv1_1);
	
	auto conv2_1 = layer::conv("conv2_1", conv1_2, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv2_2 = layer::conv("conv2_2", conv2_1, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto pool2 = layer::maxpool("pool2", conv2_1);

	auto conv3_1 = layer::conv("conv3_1", conv2_2, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_2 = layer::conv("conv3_2", conv3_1, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_3 = layer::conv("conv3_3", conv3_2, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1));
	//auto pool3 = layer::maxpool("pool3", conv3_1, Shape(2,2), Shape(2,2), Shape(1,1));
	
	auto conv4_1 = layer::conv("conv4_1", conv3_2, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_2 = layer::conv("conv4_2", conv4_1, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_3 = layer::conv("conv4_3", conv4_2, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool4 = layer::maxpool("pool4", conv4_3, Shape(3,3), Shape(1,1), Shape(0,0));

	auto conv5_1 = layer::conv("conv5_1", pool4, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_2 = layer::conv("conv5_2", conv5_1,  weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_3 = layer::conv("conv5_3", conv5_2, weight, bias, aux, 1, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1), false);

	//auto pool5 = layer::maxpool("pool5", conv5_3, Shape(3,3), Shape(2,2), Shape(0,0));
	
	auto fc1 = layer::fullyconnected("fcsggxxwwfs11", conv5_2, weight, bias, 1024);
	
	auto fc1_sigmoid = sigmoid(fc1);

	auto fc2 = layer::fullyconnected("fcss22", fc1_sigmoid, weight, bias, 1);

	auto fc2_sigmoid = sigmoid(fc2);
	
	/*

	auto fc1 = layer::fullyconnected("fcsggxxwwfs11", pool5, weight, bias, 10);	
	auto fc1_sigmoid = LeakyReLU(fc1) / 10;

	auto fc2 = layer::fullyconnected("fcss22", fc1_sigmoid, weight, bias, 10);
	auto fc2_sigmoid = LeakyReLU(fc2) / 10;
	
	auto fc3 = layer::fullyconnected("fcss33", fc2_sigmoid, weight, bias, 1);
	auto fc3_sigmoid = sigmoid(fc3);
	
	*/

	return sigmoid(conv5_3);


}

inline void network::DEVGG16_Deprecated(char * p, char * inputs, map <string, Symbol> *nodes, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){

	
	layer::deconv(p, "deconv5_3", inputs, nodes, weight, bias, 512);
	layer::deconv(p, "deconv5_2", "deconv5_3", nodes, weight, bias, 512);
	layer::deconv(p, "deconv5_1", "deconv5_2", nodes, weight, bias, 512, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool5", "deconv5_1", nodes, Shape(2,2), Shape(1,1), Shape(1,1));
	cout << "hhh" << endl;
	layer::deconv(p, "deconv4_3", "depool5", nodes, weight, bias, 512);
	layer::deconv(p, "deconv4_2", "deconv4_3", nodes, weight, bias, 512);
	layer::deconv(p, "deconv4_1", "deconv4_2", nodes, weight, bias, 256, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool4", "deconv4_1", nodes, Shape(2,2), Shape(1,1));

	layer::deconv(p, "deconv3_3", "depool4", nodes, weight, bias, 256);
	layer::deconv(p, "deconv3_2", "deconv3_3", nodes, weight, bias, 256);
	layer::deconv(p, "deconv3_1", "deconv3_2", nodes, weight, bias, 128, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool3", "deconv3_1", nodes, Shape(2,2), Shape(1,1));

	layer::deconv(p, "deconv2_2", "depool3", nodes, weight, bias, 128);
	layer::deconv(p, "deconv2_1", "deconv2_2", nodes, weight, bias, 64, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool2", "deconv2_1", nodes, Shape(2,2), Shape(1,1));
	
	layer::deconv(p, "deconv1_2", "depool2", nodes, weight, bias, 64, Shape(3,3), Shape(1,1));
	layer::deconv(p, "generated", "deconv1_2", nodes, weight, bias, 3, Shape(3,3), Shape(2,2));
	
	//layer::maxpool("depool1", (*nodes)[string("deconv1_1")], nodes, Shape(2,2), Shape(1,1), Shape(-1,-1));
	//layer::deconv("generated", (*nodes)["depool1"], nodes, weight, bias, 3);

}


inline Symbol network::DEVGG16(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, map <string, Symbol> * aux, cv::Size size){

	bool isTraining = true;

	auto deconv_en_1 = layer::conv("deconv_en_1_custom", (*inputs), weight, bias, aux, 64, "bn", isTraining);
	
	auto pool1 = layer::maxpool("pool1", deconv_en_1, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_2 = layer::conv("deconv_en_2_custom", pool1, weight, bias, aux, 128, "bn", isTraining);

	auto pool2 = layer::maxpool("pool2", deconv_en_2, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_3 = layer::conv("deconv_en_3_custom", pool2, weight, bias, aux, 256, "bn", isTraining);

	auto pool3 = layer::maxpool("pool3", deconv_en_3, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_4 = layer::conv("deconv_en_4_custom", pool3, weight, bias, aux, 512, "bn", isTraining);

	auto pool4 = layer::maxpool("pool4", deconv_en_4, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_5 = layer::conv("deconv_en_5_custom", pool4, weight, bias, aux, 512, "bn", isTraining);
	
	auto pool5 = layer::maxpool("pool5", deconv_en_5, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_6 = layer::conv("deconv_en_6_custom", pool5, weight, bias, aux, 512, "bn", isTraining);

	auto pool6 = layer::maxpool("pool6", deconv_en_6, Shape(2,2), Shape(2,2), Shape(0,0));
	
	auto deconv_en_7 = layer::conv("deconv_en_7_custom", pool6, weight, bias, aux, 512, "bn", isTraining);

	auto pool7 = layer::maxpool("pool7", deconv_en_7, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_en_8 = layer::conv("deconv_en_8_custom", pool7, weight, bias, aux, 512, "bn", isTraining);

	auto pool8 = layer::maxpool("pool8", deconv_en_8, Shape(2,2), Shape(2,2), Shape(0,0));

	auto deconv_de_1 = layer::deconv("deconv_de_1_custom", pool8, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));

	auto deconv_de_2 = layer::deconv("deconv_de_2_custom", deconv_de_1, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_3 = layer::deconv("deconv_de_3_custom", deconv_de_2, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));

	auto deconv_de_4 = layer::deconv("deconv_de_4_custom", deconv_de_3, weight, bias, aux, 512, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));

	auto deconv_de_5 = layer::deconv("deconv_de_5_custom", deconv_de_4, weight, bias, aux, 256, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_6 = layer::deconv("deconv_de_6_custom", deconv_de_5, weight, bias, aux, 128, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_7 = layer::deconv("deconv_de_7_custom", deconv_de_6, weight, bias, aux, 64, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv_de_8 = layer::deconv("deconv_de_8_custom", deconv_de_7, weight, bias, aux, 3, "bn", isTraining, Shape(3,3), Shape(2,2), Shape(1,1), Shape(1,1), Shape(1,1), false);

	/*
	auto deconv5_3 = layer::deconv("deconv5_3_custom", (*inputs), weight, bias, 32);
	auto deconv5_2 = layer::deconv("deconv5_2_custom", deconv5_3, weight, bias, 32);
	auto deconv5_1 = layer::deconv("deconv5_1_custom", deconv5_2, weight, bias, 64);
	
	auto deconv4_3 = layer::deconv("deconv4_3_custom", deconv5_1, weight, bias, 64);
	auto deconv4_2 = layer::deconv("deconv4_2_custom", deconv4_3, weight, bias, 64);
	auto deconv4_1 = layer::deconv("deconv4_1_custom", deconv4_2, weight, bias, 128);
	
	auto deconv3_3 = layer::deconv("deconv3_3_custom", deconv4_1, weight, bias, 128);
	auto deconv3_2 = layer::deconv("deconv3_2_custom", deconv3_3, weight, bias, 64);
	auto deconv3_1 = layer::deconv("deconv3_1_custom", deconv3_2, weight, bias, 64);	
	auto deconv2_2 = layer::deconv("deconv2_2_custom", deconv3_1, weight, bias, 32);
	auto deconv2_1 = layer::deconv("deconv2_1_custom", deconv2_2, weight, bias, 32);
	*/
	
	
	//auto deconv1_1 = layer::deconv("deconv1_1_custom", deconv2_1, weight, bias, aux, 3, "bn", isTraining, Shape(3,3), Shape(1,1), Shape(1,1), Shape(), Shape(), false);

	return sigmoid(deconv_de_8);
}

inline Symbol network::MLP(Symbol *inputs, Symbol *condition, int nbatch, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){
	
	vector <Symbol> features_vector {(*inputs), (*condition)};

	auto features = layer::concat("features", &features_vector,  1);
	
	auto fc1 = layer::fullyconnected("fc1", features, weight, bias, 128);
	
	auto relu1 = relu(fc1);

	auto fc2 = layer::fullyconnected("fc2", relu1, weight, bias, 1);
	
	return sigmoid(fc2); // non linear activation function


}

inline Symbol network::DEMLP(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, cv::Size size){

	auto fc1 = layer::fullyconnected("defc1", (*inputs), weight, bias, 128);
	
	auto relu1 = relu(fc1);

	auto fc2 = layer::fullyconnected("defc2", relu1, weight, bias, 784);
	
	return sigmoid(fc2); // linear (regression)
	
}


#endif
