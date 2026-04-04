# 1. Use a standard Python image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (needed for some AI libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your project files
COPY . /app

# 5. Install Python dependencies
# We use 'python -m pip' to ensure it's linked to the correct python version
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# 6. Expose the port Streamlit uses
EXPOSE 7860

# 7. The most important part: Use 'python -m streamlit' 
# This avoids the "executable not found" error by calling it through python
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]