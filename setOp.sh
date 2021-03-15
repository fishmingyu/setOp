nvcc -std=c++11 -O3 -ftz=true -arch=sm_75 -fmad=true -prec-div=true -c setOp.cu
nvcc -o setOp setOp.o
rm setOp.o