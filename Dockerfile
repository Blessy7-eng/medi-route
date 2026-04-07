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

# 4. Copy your project files
# Make sure your requirements.txt is in the root directory
COPY . /app

# 5. Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip
# We explicitly add uvicorn here to ensure the Streamlit ASGI server starts
RUN python -m pip install --no-cache-dir uvicorn fastapi streamlit
RUN python -m pip install --no-cache-dir -r requirements.txt

# 6. Expose the port Hugging Face expects
EXPOSE 7860

# 7. Start the app
# Using "streamlit run" is the standard for HF Spaces
CMD ["python", "server.py"]