#!/bin/bash
# 1. Run the AI logic (inference.py) to generate the required [START]/[END] logs.
# This script must NOT start a server/uvicorn.
python inference.py

# 2. Only AFTER inference is done, start the UI on the expected port.
streamlit run app.py --server.port 7860 --server.address 0.0.0.0