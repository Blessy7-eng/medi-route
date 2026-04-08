FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Upgrade pip and install requirements
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir uvicorn fastapi streamlit -r requirements.txt

# Ensure start.sh is executable and has Linux line endings
RUN dos2unix start.sh && \
    sed -i 's/\r$//' start.sh && \
    chmod +x start.sh

EXPOSE 7860

# Start via the shell script
CMD ["./start.sh"]