#!/usr/bin/env bash

function check_python {
python3 -c "import torch, torch_optimizer, numpy, cython, flatland, gym, tqdm; print('Successfully imported packages')" 2>/dev/null\
|| (python3.7 -m venv venv && source venv/bin/activate && python3 -m pip install -r requirements.txt) || check_python
}

check_python

echo "Checking for GCC-7"

gcc-7 --version > /dev/null || sudo apt install gcc-7

echo "Compiling source"
cd src && source cythonize.sh > /dev/null 2>/dev/null
cd ..