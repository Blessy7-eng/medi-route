# 1. Use a standard Python image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your project files (including uv.lock and app.py)
COPY . /app

# 5. Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip

# Combined installation to save time and layers
RUN python -m pip install --no-cache-dir \
    uvicorn \
    fastapi \
    streamlit \
    -r requirements.txt

# 6. Expose the port Hugging Face expects
EXPOSE 7860

# 7. Start both the UI and the Inference API
# We run Streamlit in the background on a different port, 
# and let inference.py own the main port 7860 for the validator.
CMD streamlit run app.py --server.port 8501 --server.address 0.0.0.0 & python inference.py