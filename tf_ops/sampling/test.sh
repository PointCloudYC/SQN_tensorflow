#/bin/bash
CUDA_ROOT="/usr/local/cuda-10.1"
TF_ROOT="/home/yinchao/miniconda3/envs/DL/lib/python3.6/site-packages/tensorflow"

${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC