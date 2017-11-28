CC = g++


# Set the mxnet source directory

MXNET = ~/src/mxnet

MXNET_LIB = ${MXNET}/lib

LIBRARY = -L ${MXNET_LIB} \
-l boost_system \
-l boost_filesystem \
-l boost_program_options \
-l protobuf \
-l opencv_highgui \
-l opencv_imgcodecs \
-l opencv_imgproc \
-l opencv_core \
-l lmdb \
-l mxnet \
-l hdf5 \
-l hdf5_cpp \
-l hdf5_hl \
-l pthread \
-l leveldb
#CC_FILE = io.cc  kvstore.cc  ndarray.cc  operator.cc  optimizer.cc  symbol.cc



INCLUDE = -I ./ \
-I /usr/local/include \
-I ${MXNET}/include \
-I ${MXNET}/nnvm/include \
-I ${MXNET}/dmlc-core/include \
-I flt/include \
-I flt

DEPENDENCIES = main.cc pix2pix.hh pix2pix.h init.hh init.h loss.hh init.hh init.h data.hh data.h network.hh network.h flt/src/file.hh flt/src/mx/layer.hh flt/src/mx/shape.hh flt/src/stl.hh flt/src/debug.hh flt/src/matrix.hh flt/src/cv.hh flt/src/mx/image.hh flt/src/mx/image.h

main: ${DEPENDENCIES}

	${CXX} ${INCLUDE} ${LIBRARY} main.cc -o main -Wno-narrowing -Wno-write-strings

mnist: main_mnist.cc loss.hh init.hh init.h cgan_mnist.hh cgan_mnist.h network.hh network.h flt/src/file.hh flt/src/mx/layer.hh flt/src/mx/shape.hh flt/src/stl.hh flt/src/debug.hh flt/src/matrix.hh flt/src/cv.hh flt/src/mx/image.hh flt/src/mx/image.h



	${CXX} ${INCLUDE} ${LIBRARY} main_mnist.cc annotationdb.pb.cc -o mnist -Wno-narrowing -Wno-write-strings

mnist_simple: main_mnist_simple.cc loss.hh init.hh init.h cgan_mnist_simple.hh cgan_mnist_simple.h network.hh network.h flt/src/file.hh flt/src/mx/layer.hh flt/src/mx/shape.hh flt/src/stl.hh flt/src/debug.hh flt/src/matrix.hh flt/src/cv.hh flt/src/mx/image.hh flt/src/mx/image.h



	${CXX} ${INCLUDE} ${LIBRARY} main_mnist_simple.cc annotationdb.pb.cc -o mnist_simple -Wno-narrowing -Wno-write-strings
test: test.cc

	${CXX} test.cc ${LIBRARY} ${INCLUDE} -Wno-narrowing -Wno-write-strings

tt: cgan.hh
	
	${CXX} -shared cgan.hh ${LIBRARY} ${INCLUDE}

p: prototest.cc
	
	${CXX} prototest.cc annotationdb.pb.cc ${LIBRARY} ${INCLUDE}

ppp: test_lib.cc
	
	${CXX} test_lib.cc ${LIBRARY} ${INCLUDE}
