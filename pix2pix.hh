#ifndef PIX2PIX_HH
#define PIX2PIX_HH

#include <map>
#include <cmath>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include "pix2pix.h"
#include "loss.hh"
#include "init.hh"
#include "data.hh"
#include "config.h"
#include "network.hh"

using namespace std;
using namespace mxnet::cpp;
using namespace flt::fmx;



pix2pix::pix2pix(config c){

	sdataset = c.sdataset;

	slist = c.slist;

	label = c.label;

	size = c.size;

	nbatch = c.nbatch;

	device = c.device;

	nobject = c.nobject;
	
	nnoise = c.nnoise;

	nclass = label.size();

	pretrained = c.pretrained;

	checkpoint = c.checkpoint;

};


inline void pix2pix::build(){			
	
	cout << "Start Build" << endl;

	node["generated"] = network::UNet(&c, &weight, &bias, &aux, size);
	
	node["cc"] = c;

		
	vector <Symbol> decision_real = network::pix2pix_D(&inputs, &c, nbatch, &weight, &bias, &aux, size); // presigmoid
	
		
	vector <Symbol> decision_fake = network::pix2pix_D(&node["generated"], &c, nbatch, &weight, &bias, &aux, size); // presigmoid
	
	node["decision_fake_patch"] = decision_fake[0];

	//node["decision_fake"] = decision_fake[1];

	node["decision_real_patch"] = decision_real[0];
	
	//node["decision_real"] = decision_real[1];

};

inline Symbol pix2pix::G_Loss(){

	/* let generated image can be considered a real image, so use ones_like, not zeros_like */

	//auto generation_loss = mean(loss::cross_entropy(node["decision_fake"], ones_like("generate_loss", node["decision_fake"])));
	
	auto generation_loss_patch = mean(loss::cross_entropy(node["decision_fake_patch"], ones_like("generate_loss_patch", node["decision_fake_patch"])));
	
	auto reconstruction_loss = mean(loss::L1(node["generated"], inputs));

	return MakeLoss("G_Loss", generation_loss_patch + 100 * reconstruction_loss);
}


inline Symbol pix2pix::D_Loss(){	

	//Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],
	//			ones_like("real_loss", node["decision_real"])));
	
	//Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],
	//			zeros_like("fake_loss", node["decision_fake"])));

	Symbol real_decision_loss_patch = mean(loss::cross_entropy(node["decision_real_patch"],
				ones_like("real_loss_patch", node["decision_real_patch"])));
	
	Symbol fake_decision_loss_patch = mean(loss::cross_entropy(node["decision_fake_patch"],
				zeros_like("fake_loss_patch", node["decision_fake_patch"])));
	return MakeLoss("D_Loss", real_decision_loss_patch + fake_decision_loss_patch);
}


inline void pix2pix::train(int epoch, int device_id){
	
	Context ctx(device, device_id);
	
	data::db dataset(pix2pix::sdataset, pix2pix::slist, pix2pix::label, pix2pix::size, pix2pix::nbatch, true, data::MODE::generation);

	/* setup shape */

	map <string, map <string, vmx_shape>> inf; // store infered shape
	
	map <string, map <string, mx_shape>> arg; // input shape

	arg["e"]["inputs"] = {nbatch, size.height, size.width, 3};

	arg["e"]["c"] = {nbatch, size.height, size.width, 3};
	
	arg["d"] = arg["e"];

	arg["g"]["c"] = {nbatch, size.height, size.width, 3};
	
	map <string, NDArray> nd, grad, ndaux;
	
	Uniform uniform(0, 1);

	Symbol g = G_Loss(); Symbol d = D_Loss(); Symbol e = g + d;

	node["generated"].InferShape(arg["g"], &inf["t"]["in"], &inf["t"]["aux"], &inf["t"]["out"]);

	cout << "Generated Image : " << Shape(inf["t"]["out"][0]) << endl;
	
	e.InferShape(arg["e"], &inf["e"]["in"], &inf["e"]["aux"], &inf["e"]["out"]);
	
	d.InferShape(arg["d"], &inf["d"]["in"], &inf["d"]["aux"], &inf["d"]["out"]);
	
	g.InferShape(arg["g"], &inf["g"]["in"], &inf["g"]["aux"], &inf["g"]["out"]);
	
	node["decision_real_patch"].InferShape(arg["e"], &inf["test"]["in"], &inf["test"]["aux"], &inf["test"]["out"]);

	cout << "D Shape : " << Shape(inf["test"]["out"][0]) << endl;

	//node["generated"].InferShape(gs, &gen_in, &gen_aux, &gen_out);

	//cout << "Generated Image : " << Shape(gen_out[0]) << endl;
	
	map <string, vector <string>> nnode;
	
	map <string, vector <string>> naux;

	nnode["e"] = e.ListArguments();
	
	nnode["g"] = g.ListArguments();

	nnode["d"] = d.ListArguments();
	
	naux["e"] = e.ListAuxiliaryStates();
	
	naux["g"] = g.ListAuxiliaryStates();
	
	naux["d"] = d.ListAuxiliaryStates();

	for (auto &i : inf["g"]["aux"])

		cout << "aux : " << Shape(i) << endl;
	
	/* setup weight */

	/* Deprecated */
	
	cout << "init " << endl;
	
	string weight_file;
	
	init::init_mode modes;

	int iters_chkp;

	if (checkpoint != ""){
		
		modes = init::init_mode::restore;

		weight_file = checkpoint;

		vector <string> strs;
		
		boost::split(strs, checkpoint, boost::is_any_of("/."));

		iters_chkp = stoi(strs[1]);

	}

	else{ 

		modes = init::init_mode::pretrained;

		weight_file = pretrained;

		iters_chkp = 0;

	}
	
	cout << "init weight " << endl;
	
	init::init_weight_simple(nnode["e"], inf["e"]["in"], nd, grad, ctx, modes, weight_file);
	
	init::init_aux(naux["e"], inf["e"]["aux"], ndaux, ctx, modes, weight_file);

	dataset.next();
	
	cout << "Next" << endl;
	
	cout << dataset.inputs[0].size().width << endl;
	
	cout << dataset.inputs[0].size().height << endl;
	
	cout << dataset.inputs[0].channels() << endl;

	cout << dataset.inputs.size() << endl;
	
	//nd["inputs"] = NDArray (Shape(1, 256, 256, 3), ctx);

	//nd["c"] = NDArray (Shape(1, 256, 256, 3), ctx);
	//
	//

	
	nd["inputs"] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);		

	nd["c"] = fimage::MatVector_to_NDArray(dataset.target, ctx);

	/* setup executor */

	map <string, OpReqType> opreq;

	/* setup gradient operation */


	
	/* Sample with aux */

	Executor * S = fimage::decodeb(node["generated"]).SimpleBind(ctx, nd, grad, opreq, ndaux);
	
	Executor * G = g.SimpleBind(ctx, nd, grad, opreq, ndaux);
	
	Executor * D = d.SimpleBind(ctx, nd, grad, opreq, ndaux);
	
	//vector <Symbol> groups {g, d};


//	cout << (*G).DebugStr() << endl;

//	cout << (*D).DebugStr() << endl;

	//Symbol group = Symbol::Group(groups);
	
	//Executor * E = d.SimpleBind(ctx, nd, grad);
	//
	
 	Optimizer * Gadam = OptimizerRegistry::Find("adam");
 	
	Optimizer * Dadam = OptimizerRegistry::Find("adam");

	Gadam->SetParam("lr", 0.00005);
//		->SetParam("clip_gradient", 0.05);
	
	Dadam->SetParam("lr", 0.0001);
//		->SetParam("clip_gradient", 0.05);
	
	bool dswitch = true;

	bool gswitch = true;

	vector <float> d_loss(1, 10);

	vector <float> g_loss(1, 10);

	float d_loss_obj = 0.45;

	float g_loss_obj = 1;


	for (long int i = iters_chkp; i != epoch; ++i){
		
		//(*S).Forward(false);

		//cout << ((*S).outputs[0].Reshape(Shape(256 * 256 * 3)).Slice(0,5)) << endl;
		(*D).Forward(true);
		
		if (d_loss[0] > d_loss_obj and i != iters_chkp){
			
			(*D).Backward();
					
			for (int j = 0; j != (*D).arg_arrays.size(); ++j){
					
				string prefix = nnode["d"][j].substr(0,4);

				string prefix7 = nnode["g"][j].substr(0,7);
				
				if ((prefix == "wcon") or (prefix == "bcon") or (prefix == "w_fc") or (prefix == "b_fc") or (prefix == "gammaco") or (prefix == "betacon")){

					//cout << "D -> Update : " <<  nnode["d"][j] << (*D).grad_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;
					
					(*Dadam).Update(j, (*D).arg_arrays[j], (*D).grad_arrays[j]);
				}
			}
		}

		(*G).Forward(true);
		
		if (g_loss[0] > g_loss_obj and i != iters_chkp){
			
			(*G).Backward();	
					
			for (int j = 0; j != (*G).arg_arrays.size(); ++j){
					
				string prefix = nnode["g"][j].substr(0,4);
				
				string prefix7 = nnode["g"][j].substr(0,7);

				//cout << nnode["g"][j] << endl;
				
				if ((prefix == "wdec") or (prefix == "bdec") or (prefix == "w_de") or (prefix == "b_de") or (prefix7 == "gammade") or (prefix7 == "betadec")){
					//cout << nnode["g"][j] << endl;
					//cout << "G -> Update : " <<  nnode["g"][j] << (*G).grad_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;

					(*Gadam).Update(j, (*G).arg_arrays[j], (*G).grad_arrays[j]);


				}	
			}
		}
		
		/*for (auto &i : aux){
			
			cout << i.first << " : " << i.second << endl;

		}*/

		vector <NDArray> dout = (*D).outputs;
	
		vector <NDArray> gout = (*G).outputs;
		
		dout[0].SyncCopyToCPU(d_loss.data(), 1);
		
		gout[0].SyncCopyToCPU(g_loss.data(), 1);
		
		if (i % 1 == 0){

			cout << "Epoch : " << i << ", D Loss : " << d_loss[0] << ", G Loss : " << g_loss[0];
			
			cout << ", D : ";

		if (d_loss[0] >= d_loss_obj)

				cout << "On";

			else

				cout << "Off";
			
			cout << ", G : ";

			if (g_loss[0] >= g_loss_obj)
				
				cout << "On";
			else

				cout << "Off";


			cout << endl;
				
		}

		if (d_loss[0] <= d_loss_obj and g_loss[0] <= g_loss_obj){

			d_loss_obj /= 2.0;

			g_loss_obj /= 2.0;
		}

		else if (isnan(d_loss[0]) or isnan(g_loss[0])){
			
			cout << "[!] NAN ... " << endl;
			
			break;
		}
		
			
		
		if (i % 500 == 0){
			
			//uniform("z", &nd["z"]);
			
			vector <string> sample_list = {"100.jpg", "200.jpg", "300.jpg", "400.jpg", "500.jpg"};


		
			
			try{
				
				for (int ss = 0; ss != sample_list.size(); ++ss){

					vector <cv::Mat> sm;
					
					cv::Mat cvimage = cv::imread(sample_list[ss], cv::IMREAD_COLOR);
					
					cv::Mat inrimage(256, 256, CV_8UC3);
						
					cv::Mat outrimage(256, 256, CV_8UC3);

					cv::Rect inROI(0, 0, 256, 256);

					cv::Rect outROI(256, 0, 256, 256);

					inrimage = cvimage(inROI);

					outrimage = cvimage(outROI);

					sm.emplace_back(outrimage);

					fimage::MatVector_to_NDArray(nd["c"], sm, ctx);
			
					(*S).Forward(false);
		
					vector <NDArray> sout = (*S).outputs;

					sm.shrink_to_fit();
					
					NDArray fig = sout[0].Slice(0, 1);

					cout << "sout[" << ss << "] :" << Shape(sout[0].GetShape()) << endl;

					cout << "nd['c'] : " << Shape(nd["c"].GetShape()) << endl;
					
					fimage::save("patch_scalar_rt_by_ce/" + sample_list[ss].substr(0,3) + ".jpg", fig, 255);

					fimage::save("patch_scalar_rt_by_ce/" + sample_list[ss].substr(0,3) + "_label.jpg", nd["c"], 255);
					
					//fimage::save("patch_scalar_rt_by_ce" + to_string(i) + "_in.jpg", nd["inputs"], 255);
					cv::imwrite("patch_scalar_rt_by_ce/" + sample_list[ss].substr(0,3) + "_in.jpg", inrimage);
				}
				
				//cout << "nd['c']" << nd["c"] << endl;
				
				//cv::imwrite("out/" + to_string(i) + "_in.jpg", dataset.inputs[0]);
				
				//cv::imwrite("out/" + to_string(i) + "_label.jpg", dataset.target[0]);
				
				//cout << "File : " << dataset.proposals_key[dataset.batch[0]] << endl;

			}

			catch (...){

				cout << "Error occur when writing Image ... " << endl;
			}


		}

		//cout << d_loss[0] << endl;
		//cout << "G Loss : " << gout[0] << endl;
		
		//cout << "S out : " << Shape(sout[0].GetShape()) << endl;
		
		
		//fimage::saveb("out/" + to_string(i), i, nd_d[input_d]);
	
		dataset.next();
		

		/* Cause Memory probelm  */

		fimage::MatVector_to_NDArray(nd["inputs"], dataset.inputs, ctx);		
		
		fimage::MatVector_to_NDArray(nd["c"], dataset.target, ctx);
		
		//nd["c"] = fimage::MatVector_to_NDArray(dataset.target, ctx);

		//fimage::MatVector_to_NDArray(nd["c"], dataset.target, ctx);
		
		if (i % 5000 == 0){
			
			fhdf5 saver("model/" + to_string(i) + ".chk");

			saver.open();
			
			saver.save_NDArray(nd);

			saver.save_NDArray(ndaux);

			//saver.save_NDArray(grad);
			
			saver.close();

		}
	}
	
	delete G;

	delete D;

	delete S;

	MXNotifyShutdown();

};

pix2pix::~pix2pix(){
	
//	MXNotifyShutdown();

}

inline void pix2pix::test(string image){
	
	Context ctx(device, 0);
	
	//data::db dataset(pix2pix::sdataset, pix2pix::slist, pix2pix::label, pix2pix::size, pix2pix::nbatch, true, data::MODE::generation);

	/* setup shape */

	map <string, map <string, vmx_shape>> inf; // store infered shape
	
	map <string, map <string, mx_shape>> arg; // input shape

	arg["e"]["inputs"] = {nbatch, size.height, size.width, 3};

	arg["e"]["c"] = {nbatch, size.height, size.width, 3};
	
	arg["d"] = arg["e"];

	arg["g"]["c"] = {nbatch, size.height, size.width, 3};
	
	map <string, NDArray> nd, grad, ndaux;
	
	Uniform uniform(0, 1);

	Symbol g = G_Loss(); Symbol d = D_Loss(); Symbol e = g + d;

	node["generated"].InferShape(arg["g"], &inf["t"]["in"], &inf["t"]["aux"], &inf["t"]["out"]);

	//cout << "Generated Image : " << Shape(inf["t"]["out"][0]) << endl;
	
	e.InferShape(arg["e"], &inf["e"]["in"], &inf["e"]["aux"], &inf["e"]["out"]);
	
	d.InferShape(arg["d"], &inf["d"]["in"], &inf["d"]["aux"], &inf["d"]["out"]);
	
	g.InferShape(arg["g"], &inf["g"]["in"], &inf["g"]["aux"], &inf["g"]["out"]);
	
	node["decision_real"].InferShape(arg["e"], &inf["test"]["in"], &inf["test"]["aux"], &inf["test"]["out"]);

	//cout << "D Shape : " << Shape(inf["test"]["out"][0]) << endl;

	//node["generated"].InferShape(gs, &gen_in, &gen_aux, &gen_out);

	//cout << "Generated Image : " << Shape(gen_out[0]) << endl;
	
	map <string, vector <string>> nnode;
	
	map <string, vector <string>> naux;

	nnode["e"] = e.ListArguments();
	
	nnode["g"] = g.ListArguments();

	nnode["d"] = d.ListArguments();
	
	naux["e"] = e.ListAuxiliaryStates();
	
	naux["g"] = g.ListAuxiliaryStates();
	
	naux["d"] = d.ListAuxiliaryStates();

	//for (auto &i : inf["g"]["aux"])

	//	cout << "aux : " << Shape(i) << endl;
	
	/* setup weight */

	/* Deprecated */
	
	string weight_file;
	
	init::init_mode modes;

	int iters_chkp;

	if (checkpoint != ""){
		
		modes = init::init_mode::predict;

		weight_file = checkpoint;

		vector <string> strs;
		
		boost::split(strs, checkpoint, boost::is_any_of("/."));

		iters_chkp = stoi(strs[1]);

	}

	else

		cout << "Test Image is not set" << endl;

	init::init_weight_simple(nnode["e"], inf["e"]["in"], nd, grad, ctx, modes, weight_file);
	
	init::init_aux(naux["e"], inf["e"]["aux"], ndaux, ctx, modes, weight_file);


	cv::Mat cvimage = cv::imread(image, cv::IMREAD_COLOR);
	
	cv::Mat inrimage(256, 256, CV_8UC3);
		
	cv::Mat outrimage(256, 256, CV_8UC3);

	cv::Rect inROI(0, 0, 256, 256);

	cv::Rect outROI(256, 0, 256, 256);

	inrimage = cvimage(inROI);

	outrimage = cvimage(outROI);

	
	nd["c"] = fimage::Mat_to_NDArray(outrimage, ctx).Reshape(Shape(1, 256, 256, 3));
	
	map <string, OpReqType> opreq;
	
	/* Sample with aux */

	Executor * S = fimage::decodeb(node["generated"]).SimpleBind(ctx, nd, grad, opreq, ndaux);
	
	(*S).Forward(false);
	
	vector <NDArray> sout = (*S).outputs;

	NDArray fig = sout[0].Slice(0, 1);

	try{

		fimage::save("test/" + image + ".jpg", fig, 255);

		fimage::save("test/" + image + "_label.jpg", nd["c"], 255);
	
	}

	catch (...){

		cout << "Error occur when writing Image ... " << endl;
	}


	
	delete S;

	MXNotifyShutdown();

};
#endif
