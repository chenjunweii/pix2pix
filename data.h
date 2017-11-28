#ifndef DATA_NORMAL_H
#define DATA_NORMAL_H

#include <iostream>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>
#include <rapidxml/rapidxml_print.hpp>

#include <algorithm>

using namespace std;

namespace data{
	
	enum MODE { detection, generation };

	struct annotation{

		int x;
		int y;
		int w;
		int h;
		int c; // specify the index of class of object , ex. 2 ;
		int * i; // class array of object , ex. [0,0,1,0,0]
		vector <string> vs; // for generative model


	};

	struct SIZE{

		int w;
		int h;

	};

	struct RATIO{

		float w;
		float h;
	};

	struct cBox{

		/*

		x, y -> center of box

		w, h;

		*/

		int x;
		int y;
		int w;
		int h;
	};

	struct ulBox{

		/*

		x, y -> upper left of the box

		w, h

		*/

		int xmin;
		int ymin;
		int w;
		int h;

	};

	struct bBox{

		/*

		x, y -> boundary of box

		*/

		int xmin;
		int ymin;
		int xmax;
		int ymax;

	};



	class db{

		public:

			string sdataset, basedir, annotationdir;

			char * proposal;

			int n_image_entry = 1; // image entry

			int n_proposal_entry = 1;

			int shuffle_counter; // shuffle when counter goes to zero;

			int width, height;

			int n_class;

			int batch_size;

			int step = 0;

			bool DEBUG;
			
			MODE mode;

			leveldb::Options options;
			
			leveldb::DB * env;

			vector <int> batch; // stores the randomly select batch;

			vector <int> proposals_batch; // for counter and "next function"
			
			vector <int> proposals_key; // real key stored in lmdb, for random
			
			vector <string> filenames;
			
			/*
			 *	proposals stay the same, it is used for next function to 
			 *
			 *	caluculate which batch to use
			 *
			 *	but we shuffle the index of real key in "proposals for random"
			 *
			 *
			 *
			 */
			
			vector <string> label;

			vector <cv::Mat> inputs; // vector stores the loaded image

			vector <cv::Mat> target; // vector stores the loaded annotation
			


			/* function declare */

			db(string, string, vector <string>, cv::Size, int, bool, MODE); // contructor

			inline int next();

			inline int load(vector <int>);

			inline int merge(string);

			inline int merge_generative(string);

			inline int shuffle();

			//int initialize();

			inline int generate(string);

			inline void generate_proposal_list();

			inline void generate_label_array(int *v, int index);




		private:

			vector <int> _proposal_list; // for random shuffle the training batch

			int _begin, _end;

			inline int _get_index_by_class(string);

	}; /* db */

} /* data */

#endif
