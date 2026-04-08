#!/bin/bash
# Run the inference first to generate logs for the grader
python inference.py

# Then start Streamlit on the required port
streamlit run app.py --server.port 7860 --server.address 0.0.0.0