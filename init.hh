#ifndef INIT_HH
#define INIT_HH

#include <map>
#include <assert.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <boost/program_options.hpp>

#include "init.h"

using namespace std;
using namespace mxnet::cpp;
using namespace flt;

namespace bpo = boost::program_options;


inline void init::init_aux(vector <string> &aux,
		vector <vector <mx_uint>> &shape,
		map <string, NDArray> &ndaux,
		Context ctx,
		init_mode im,
		string filename){
	
	assert(aux.size() == shape.size());
	
	One one;
	
	Zero zero;

		
	if (im == init_mode::predict){
		
		fhdf5 f;

		if (filename != ""){

			f = fhdf5 (filename.c_str(), &ctx);

			f.read();

			f.load_weight();
		}

		for (int i = 0; i != aux.size(); ++i){
			
			ndaux[aux[i]] = f.nddata[aux[i]];
			
			cout << ndaux[aux[i]] << endl;

		}
	}

	else{
	
		for (int i = 0; i != aux.size(); ++i){
			
			ndaux[aux[i]] = NDArray(Shape(shape[i]), ctx);

			string prefix4 = aux[i].substr(0,3);

			if (prefix4 == "var")

				zero(aux[i], &ndaux[aux[i]]);


			else if (prefix4 == "mea")

				zero(aux[i], &ndaux[aux[i]]);

		}
	
	}
}

inline void init::init_weight_simple(vector <string> &node,
		vector <vector <mx_uint>> &shape,
		map <string, NDArray> &ndarg,
		map <string, NDArray> &grad,
		Context ctx,
		init_mode im,
		string filename,
		vector <string> * pretrained,
		map <string, string> * mapping,
		bool node_mapping){

	//cout << "in init weight " << endl;
	map <string, string> vgg16;
	
	vgg16["conv5_3"] = "layer_29";
	vgg16["conv5_2"] = "layer_27";
	vgg16["conv5_1"] = "layer_25";

	vgg16["conv4_3"] = "layer_22";
	vgg16["conv4_2"] = "layer_20";
	vgg16["conv4_1"] = "layer_18";

	vgg16["conv3_3"] = "layer_15";
	vgg16["conv3_2"] = "layer_13";
	vgg16["conv3_1"] = "layer_11";

	vgg16["conv2_2"] = "layer_8";
	vgg16["conv2_1"] = "layer_6";

	vgg16["conv1_2"] = "layer_3";
	vgg16["conv1_1"] = "layer_1";

	vgg16["deconv5_3"] = "layer_29";
	vgg16["deconv5_2"] = "layer_27";
	vgg16["deconv5_1"] = "layer_25";

	vgg16["deconv4_3"] = "layer_22";
	vgg16["deconv4_2"] = "layer_20";
	vgg16["deconv4_1"] = "layer_18";

	vgg16["deconv3_3"] = "layer_15";
	vgg16["deconv3_2"] = "layer_13";
	vgg16["deconv3_1"] = "layer_11";

	vgg16["deconv2_2"] = "layer_8";
	vgg16["deconv2_1"] = "layer_6";

	vgg16["deconv1_2"] = "layer_3";
	vgg16["deconv1_1"] = "layer_1";

	Xavier xavier;
	
	Zero zero;

	One one;

	if (node.size() != shape.size()){
		
		cout << "size node node not same" << endl;
		throw bad_function_call();//"size of node name vector is not same as the size of uint shape vector");

	}

	fhdf5 f;
	
	
	if (im == init_mode::pretrained){ 
		
		if (filename != ""){

			f = fhdf5 (filename.c_str(), &ctx);
			
			f.read();

			f.load_keras_all();
			
		}
		
		for(int i = 0; i != node.size(); ++i){
		
		// Check if initialize with Pretrained Weight
			
			string prefix = node[i].substr(0,1);

			string prefix4 = node[i].substr(0,4);
			
			if (node[i].find("custom") != string::npos){
				
				cout << "Custom : " << node[i] << " : " << Shape(shape[i]) << endl;

				ndarg[node[i]] = NDArray(Shape(shape[i]), ctx);
				
				grad[node[i]] = NDArray(Shape(shape[i]), ctx);
			
				if (prefix4 == "beta"){

					zero(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);

				}

				else if (prefix4 == "gamm"){

					one(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);

				}


				else if (prefix == "w"){
					
					xavier(node[i], &ndarg[node[i]]);

					zero(node[i], &grad[node[i]]);
 					
				}
				
				else if (prefix == "b"){
					
					zero(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);
				}

			}
		
			else if (node[i].find("conv") != std::string::npos) {
		
				string layer = node[i].substr(1);

				cout << "Conv : " << node[i] << " : " << Shape(shape[i]) << endl;
				
				//cout << "Layer : " << vgg16[layer] << endl;
		
				float *fp;
				
				if (prefix4 == "beta"){

					ndarg[node[i]] = NDArray(Shape(shape[i]), ctx);
					
					grad[node[i]] = NDArray(Shape(shape[i]), ctx);
						
					zero(node[i], &ndarg[node[i]]);
						
					zero(node[i], &grad[node[i]]);

				}

				else if (prefix4 == "gamm"){

					ndarg[node[i]] = NDArray(Shape(shape[i]), ctx);
				
					grad[node[i]] = NDArray(Shape(shape[i]), ctx);
					
					one(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);

				}

				else if (prefix == "w"){

					fp = f.keras["weight"][vgg16[layer]].data();

					ndarg[node[i]] = fmx::nd::FArray_to_NDArray(fp, Shape(shape[i]), ctx);
					
					grad[node[i]] = NDArray(Shape(shape[i]), ctx);
					//cout << "fp size : " << f.keras["weight"][vgg16[layer]].size() << endl;
				}
				
				else if (prefix == "b"){

					fp = f.keras["bias"][vgg16[layer]].data();

					ndarg[node[i]] = fmx::nd::FArray_to_NDArray(fp, Shape(shape[i]), ctx);
					
					grad[node[i]] = NDArray(Shape(shape[i]), ctx);
					//cout << "fp size : " << f.keras["bias"][vgg16[layer]].size() << endl;
				}
				
				else

					continue;
			}
			
			else if (node[i] != "c" and node[i] != "inputs" and node[i] != "z"){
				
				cout << "Else : " << node[i] << endl;

				ndarg[node[i]] = NDArray(Shape(shape[i]), ctx);
				
				grad[node[i]] = NDArray(Shape(shape[i]), ctx);
				
				if (prefix4 == "beta"){

					zero(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);

				}

				else if (prefix4 == "gamm"){

					one(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);

				}
				else if (prefix == "w"){
					
					xavier(node[i], &ndarg[node[i]]);

					zero(node[i], &grad[node[i]]);
 					
				}
				else if (prefix == "b"){
					
					zero(node[i], &ndarg[node[i]]);
					
					zero(node[i], &grad[node[i]]);
				}
			}

			else{

				grad[node[i]] = NDArray(Shape(shape[i]), ctx);

			}
		}
	}

	else if (im == init_mode::restore or im == init_mode::predict){

		if (filename != ""){

			f = fhdf5 (filename.c_str(), &ctx);

			f.read();

			f.load_weight();
		}

		
		for (int i = 0; i != node.size(); ++i){

			if (node[i] != "c" and node[i] != "z" and node[i] != "inputs"){
			
				ndarg[node[i]] = f.nddata[node[i]];
				
				if (im == init_mode::restore){

					grad[node[i]] = NDArray(Shape(ndarg[node[i]].GetShape()), ctx);

					zero(node[i], &grad[node[i]]);

				}

			}

		}

	}

	f.close();	

}


inline void init::init_weight(vector <string> &node,
		vector <mx_shape> &shape,
		vector <NDArray> &ndarg,
		vector <NDArray> &grad,
		Context ctx,
		init_mode im,
		string filename,
		vector <string> * pretrained,
		map <string, string> * mapping,
		bool node_mapping){

	cout << "in init weight " << endl;
	map <string, string> vgg16;
	
	vgg16["conv5_3"] = "layer_29";
	vgg16["conv5_2"] = "layer_27";
	vgg16["conv5_1"] = "layer_25";

	vgg16["conv4_3"] = "layer_22";
	vgg16["conv4_2"] = "layer_20";
	vgg16["conv4_1"] = "layer_18";

	vgg16["conv3_3"] = "layer_15";
	vgg16["conv3_2"] = "layer_13";
	vgg16["conv3_1"] = "layer_11";

	vgg16["conv2_2"] = "layer_8";
	vgg16["conv2_1"] = "layer_6";

	vgg16["conv1_2"] = "layer_3";
	vgg16["conv1_1"] = "layer_1";

	vgg16["deconv5_3"] = "layer_29";
	vgg16["deconv5_2"] = "layer_27";
	vgg16["deconv5_1"] = "layer_25";

	vgg16["deconv4_3"] = "layer_22";
	vgg16["deconv4_2"] = "layer_20";
	vgg16["deconv4_1"] = "layer_18";

	vgg16["deconv3_3"] = "layer_15";
	vgg16["deconv3_2"] = "layer_13";
	vgg16["deconv3_1"] = "layer_11";

	vgg16["deconv2_2"] = "layer_8";
	vgg16["deconv2_1"] = "layer_6";

	vgg16["deconv1_2"] = "layer_3";
	vgg16["deconv1_1"] = "layer_1";

	Xavier xavier;
	
	Zero zero;

	if (node.size() != shape.size()){
		
		cout << "size node node not same" << endl;
	
		throw bad_function_call();//"size of node name vector is not same as the size of uint shape vector");

	}

	fhdf5 f;
	
	
	if (im == init_mode::pretrained){ 
	
		if (filename != ""){

			f = fhdf5 (filename.c_str(), &ctx);
			
			f.read();

			f.load_keras_all();
			
		}
		
		for(int i = 0; i != node.size(); ++i){
		
		// Check if initialize with Pretrained Weight
			
			cout << "Init [" << node[i] << "]" << endl;
			
			string prefix = node[i].substr(0,1);
		
			if (node[i].find("conv") != std::string::npos) {
		
				string layer = node[i].substr(1);


				//cout << "Conv Node : " << node[i] << endl;
				
				//cout << "Layer : " << vgg16[layer] << endl;
		
				float * fp;
				
				if (prefix == "w"){

					fp = f.keras["weight"][vgg16[layer]].data();
					
					
					//cout << "fp size : " << f.keras["weight"][vgg16[layer]].size() << endl;
				}
				
				else if (prefix == "b"){

					fp = f.keras["bias"][vgg16[layer]].data();

					//cout << "fp size : " << f.keras["bias"][vgg16[layer]].size() << endl;
				}
				
				else{
					
					cout << "Node : " << node[i] << endl;
					
					continue;
				}

				cout << "[" << i << "] [" << node[i] << "] Shape : " << Shape(shape[i]) << endl;
				
				ndarg[i] = fmx::nd::FArray_to_NDArray(fp, Shape(shape[i]), ctx);
				//ndarg[i] = NDArray(Shape(shape[i]), ctx);
				
				//xavier(node[i], &ndarg[i]);

				grad[i] = NDArray(Shape(shape[i]), ctx);
					
				zero(node[i], &grad[i]);
			}
			
			else if (node[i] != "lp" and node[i] != "inputs" and node[i] != "z"){

				ndarg[i] = NDArray(Shape(shape[i]), ctx);
				
				grad[i] = NDArray(Shape(shape[i]), ctx);
				
				cout << "Not Conv [" << node[i] << "]" << endl;

				if (prefix == "w"){

					xavier(node[i], &ndarg[i]);

					zero(node[i], &grad[i]);
 
				}
				else if (prefix == "b"){
					
					zero(node[i], &ndarg[i]);
					
					zero(node[i], &grad[i]);
				}
			}

			else{	
				
				cout << "Else : [" << i << "] [" << node[i] << "] Shape : " << Shape(shape[i]) << endl; 
				
				if (node[i] != "inputs"){
					
					grad[i] = NDArray(Shape(shape[i]), ctx);

					zero(node[i], &grad[i]);
				}
			}
		}
	}

	else if (im == init_mode::restore or im == init_mode::predict){

		if (filename != ""){

			f = fhdf5 (filename.c_str(), &ctx);

			f.read();

			f.load_weight();
		}

		
		for (int i = 0; i != node.size(); ++i){
			
			cout << "loading ... " << endl;
			ndarg[i] = f.nddata[node[i]];

			if (im == init_mode::restore){

				grad[i] = NDArray(Shape(ndarg[i].GetShape()), ctx);

			}

		}

	}

	f.close();	

}


inline vector <OpReqType> init::wrt(char * prefix,
			vector <string> &node){
	
	vector <OpReqType> grad_wrt(node.size());
	
	string p(prefix);

	int size = p.length();

	for (int i = 0; i != node.size(); ++i){
		
		if (p == node[i].substr(1, size)){
			
			cout << "Got Gradient ["  << i << "] " << p << " : " << node[i] << endl;
			//grad_wrt[i] = OpReqType::kWriteInplace;
			grad_wrt[i] = OpReqType::kWriteTo;//OpReqType::kWriteTo;

		}

		else{
			
			cout << "No Gradient ["  << i << "] " << p << " : " << node[i] << endl;

			grad_wrt[i] = OpReqType::kNullOp;

		}
	}

	return grad_wrt;

}



inline vector <OpReqType> init::wrt(vector <char *> prefix,
			vector <string> &node){
	
	vector <OpReqType> grad_wrt(node.size());
	
	
	vector <int> size (prefix.size());
	
	for (int i = 0; i != prefix.size(); ++i)

		size[i] = string(prefix[i]).length();

	for (int i = 0; i != node.size(); ++i){

		bool isbreak = false;
		
		for (int j = 0; j != prefix.size(); ++j){

			if (string(prefix[j]) == node[i].substr(1, size[j])){
				
				grad_wrt[i] = OpReqType::kWriteTo;

				cout << "Got Gradient ["  << i << "] " << string(prefix[j]) << " : " << node[i] << endl;

				isbreak = true;

				break;
			}

		}
		
		if (!isbreak){

			cout << "No Gradient ["  << i << "] : " << node[i] << endl;

			grad_wrt[i] = OpReqType::kNullOp;

		}
		
	}

	return grad_wrt;

}
#endif
