#!/bin/bash
# 1. Run the AI logic to generate [START]/[END] logs for the grader.
# Ensure inference.py DOES NOT contain 'uvicorn.run' or 'app.run'.
python inference.py

# 2. Start the Streamlit UI ONLY after the grader logic finishes.
streamlit run app.py --server.port 7860 --server.address 0.0.0.0