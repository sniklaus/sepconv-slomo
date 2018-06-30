#!/bin/bash

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

nvcc -c -o src/SeparableConvolution_kernel.o src/SeparableConvolution_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python install.py

wget --timestamping http://content.sniklaus.com/sepconv/network-l1.pytorch
wget --timestamping http://content.sniklaus.com/sepconv/network-lf.pytorch