FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir uvicorn fastapi streamlit -r requirements.txt

# Make the script executable
RUN chmod +x start.sh

EXPOSE 7860

# Use JSON format (Exec form) to prevent the warning
CMD ["./start.sh"]