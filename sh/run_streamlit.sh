#!/bin/bash
export BNB_CUDA_VERSION=118
if [ -d "AutoGPTQ" ]; then
    streamlit run streamlitui.py --browser.gatherUsageStats False --server.port 8501 
else
    git clone -b v0.7.1-release https://github.com/AutoGPTQ/AutoGPTQ
    cd AutoGPTQ
    pip install -vvv -e .
    cd ..
    streamlit run streamlitui.py --browser.gatherUsageStats False --server.port 8501 
fi