# understanding the custom CUDA ops


# problems

## shell script reports $'\r': command not found

- reason: these shell scripts are copied from windows, need remove those '\r\n' characters(windows OS) for each line in the file
- solution: 
  - use dos2unix; `dos2unix [file]`
  - manual; remove them or create a new file, copy these liens w/o '\r\n' 

refs:
* [shell脚本执行错误 $'\r':command not found_liuxiangke0210的专栏-CSDN博客](https://blog.csdn.net/liuxiangke0210/article/details/80395707)
* [How do I fix "$'\r': command not found" errors running Bash scripts in WSL? - Ask Ubuntu](https://askubuntu.com/questions/966488/how-do-i-fix-r-command-not-found-errors-running-bash-scripts-in-wsl)

## When compiling PointNet++ compile CUDA op, it report undefined symbol: _ZTIN10tensorflow8OpKernelE

- solution: need comment `#-D_GLIBCXX_USE_CXX11_ABI=0`

```
CUDA_ROOT="/usr/local/cuda-10.1"
TF_ROOT="/home/yinchao/miniconda3/envs/DL/lib/python3.6/site-packages/tensorflow"

# TF1.13
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}lib64/ -L${TF_ROOT} -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
```

refs:
* [Undefined symbol: · Issue #48 · charlesq34/pointnet2](https://github.com/charlesq34/pointnet2/issues/48)


## When compiling PointNet++ compile CUDA op, it report /usr/bin/ld: cannot find -lcudart

- solution: symlink libcudart.so, e.g., `sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so`

refs:
* [compiling - "/usr/bin/ld: cannot find -lcudart" - Ask Ubuntu](https://askubuntu.com/questions/510176/usr-bin-ld-cannot-find-lcudart)


## PointNet++ some python files is python 2.x, e.g., print is version 2

- solution: convert python 2.x to python 3.x source code by 2to3 script

```
2to3 -2 file.py
```

refs:
* [2to3 - Automated Python 2 to 3 code translation — Python 3.10.0 documentation](https://docs.python.org/3/library/2to3.html)

## always report /usr/bin/ld: cannot find -lcudart when compiling tf_ops/3d_interpolation

- reason: some minor characters
- solution: I remove all the files under 3d_interpolation folder, then re-download the files from the github repo, and then re-edit the file.

```
#/bin/bash
CUDA_ROOT="/usr/local/cuda-10.1"
TF_ROOT="/home/yinchao/miniconda3/envs/DL/lib/python3.6/site-packages/tensorflow"

# TF1.4 (Note: -L ${TF_ROOT} should have a space in between)
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 # -D_GLIBCXX_USE_CXX11_ABI=0
```