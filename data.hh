#ifndef DATA_NORMAL_HH
#define DATA_NORMAL_HH

#include <string>
#include <iostream>
#include <assert.h>
#include <unistd.h>
#include <algorithm>
#include <leveldb/db.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "flt.h"
#include "data.h"

using namespace std;

data::db::db(string directory,
		string strain,
		vector <string> _label,
		cv::Size _size,
		int _batch_size = 1,
		bool debug = false,
		data::MODE _mode = data::MODE::detection){

	sdataset = directory;

	width = _size.width;

	height = _size.height;

	batch_size = _batch_size;

	label = _label;

	DEBUG = debug;

	mode = _mode;

	basedir = sdataset + "/";

	n_class = label.size();

	flt::ffile::fiterator fi(basedir + "train.txt");
	
	fi.next();

	while (fi.next())
	
		filenames.emplace_back(fi.line);
	
	cout << filenames.size() << endl;

	n_proposal_entry = filenames.size();

	n_image_entry = filenames.size();
	
	shuffle_counter = filenames.size();

	for(int i = 0; i != n_proposal_entry; ++i){

		proposals_batch.emplace_back(i); // batch index start from 0, because it is index, not key

		proposals_key.emplace_back(i + 1); // image entry start from 1, it is key
	}

	shuffle();
}



inline int data::db::generate(string list){
	
	n_image_entry = 0;

	n_proposal_entry = 0;

	leveldb::Status s;
	
	s = (*env).Put(leveldb::WriteOptions(), "Dataset", sdataset);
	
	assert(s.ok());

	flt::ffile::fiterator iter(basedir + list);
	
	while(iter.next()){

		string path = basedir + "Images" + "/"+ iter.line;

		cout << "[Generate Dataset] Image : " << path + ".jpg -> ";

		db::merge_generative(iter.line);

		n_image_entry += 1;
	}
	
	s = env->Put(leveldb::WriteOptions(), "Image Entry", to_string(n_image_entry));
	
	assert(s.ok());

	s = env->Put(leveldb::WriteOptions(), "Proposal Entry", to_string(n_proposal_entry));
	
	assert(s.ok());
}

inline void data::db::generate_label_array(int *v, int index){

	for(int i = 0; i != n_class; i++)

		v[i] = 0; // set all values to 0

	v[index] = 1; // set index to 1
}


inline void data::db::generate_proposal_list(){

	for(int i = 0; i != n_proposal_entry; i++)

		_proposal_list.emplace_back(i);

	//db::shuffle();

}

inline int data::db::shuffle(){

	unsigned seed = (unsigned)time(NULL); // 取得時間序列

	srand(seed);

	random_shuffle(proposals_key.begin(), proposals_key.end());

}


inline int data::db::next(){

	_begin = step * batch_size % n_proposal_entry;
	
	_end = (step + 1) * batch_size % n_proposal_entry;

	if((_begin + batch_size) > n_proposal_entry){

		batch = flt::fvector::concat(
				flt::fvector::slice(proposals_batch, _begin, n_proposal_entry),
				flt::fvector::slice(proposals_batch, 0, _end));
	}
	
	else if((_begin + batch_size) == n_proposal_entry){

		batch = flt::fvector::slice(proposals_batch, _begin, n_proposal_entry);
	}

	else{

		batch = flt::fvector::slice(proposals_batch, _begin, _end);
	}


	step += 1;

	shuffle_counter -= batch_size;
	
	//cout << batch_size << endl;;

	//cout << shuffle_counter << endl;

	if (shuffle_counter <= 0){

		flt::fdebug::log("Shuffle Proposal Entry ...", DEBUG);

		shuffle();

		shuffle_counter = n_proposal_entry;
	}


	data::db::load(batch);

}

int data::db::load(vector <int> minibatch){

	/*

	Load proposal from LMDB file and do Data Augmentation on the fly

	load proposal by given shuffled list

	batch : continuous list

	_proposal list : random shuffled list

	*/
	
	inputs.clear();

	target.clear();
	
	//vector <cv::Mat> ().swap(inputs);

	//vector <cv::Mat> ().swap(target);
	//
	inputs.shrink_to_fit();

	target.shrink_to_fit();
	
	if (batch_size != minibatch.size())

		cout << "batch size is not correct";

	for(int i = 0; i != minibatch.size(); i++){
		
		/*
		 *	proposals key is a randomize real key vector
		 *	
		 *	minibatch[i] : i = batch size, minibatch contains a continueous value 
		 *
		 *	for exmaple [ 25 - 70 ]
		 *
		 *	but 25 - 70 may map to different real key in "proposals key" each iterations
		 *
		 */
		
		string path = basedir + string("train") + string("/") + to_string(proposals_key[minibatch[i]]) + ".jpg";

		cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

		cv::Mat inrimage(height, width, CV_8UC3);
		
		cv::Mat outrimage(height, width, CV_8UC3);

		cv::Rect inROI(0, 0, 256, 256);

		cv::Rect outROI(256, 0, 256, 256);

		inrimage = image(inROI);

		outrimage = image(outROI);

		//cv::Mat infrimage(height, width, CV_32FC3);

		//cv::Mat outfrimage(height, width, CV_32FC3);
		
		//cv::resize(image, rimage, cv::Size(height, width));
		
		//cv::resize(image, rimage, cv::Size(height, width));
		
		//inrimage.convertTo(infrimage, CV_32FC3);
		
		//outrimage.convertTo(outfrimage, CV_32FC3);
		
		inputs.emplace_back(inrimage);

		//cout << inputs.capacity() << endl;

		target.emplace_back(outrimage);
		
	}
	
}

inline int data::db::_get_index_by_class(string c){

	/*

	return the index of the given class

	*/

	//cout << "label size : " << label.size() << endl;
	
	if(!label.size())

		flt::fdebug::error(string("Label size is not correct : %d", label.size()));

	for(int i = 0; i != label.size(); i++){

		if(c == label[i]) // class 0 is background;


			if(!i)

				cout << "class index error ..." << endl;

			else

				return i - 1;

		//else

			//cout << "i : " << i << endl;
	}

//	cout << "get label index .. exit" << endl;

	return -1;
}

#endif
