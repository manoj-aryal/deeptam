#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
cd ~/lmbspecialops/
mkdir build && cd build && cmake .. && make
cd ~/deeptam/tracking/examples
python example_basic.py
