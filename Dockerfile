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

# Fix line endings (just in case you're on Windows) and set permissions
RUN apt-get update && apt-get install -y dos2unix && \
    dos2unix start.sh && \
    chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]