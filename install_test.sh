#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
cd /home/docker/lmbspecialops/
mkdir build && cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/samples -D_GLIBCXX_USE_CXX11_ABI=0 \
	-DGENERATE_KEPLER_SM35_CODE=ON -DGENERATE_KEPLER_SM37_CODE=ON -DGENERATE_MAXWELL_SM52_CODE=ON \
	-DGENERATE_PASCAL_SM60_CODE=ON -DGENERATE_PASCAL_SM61_CODE=ON -DGENERATE_PTX61_CODE=ON .. && make
cd /home/docker/deeptam/tracking/examples
python example_basic.py
