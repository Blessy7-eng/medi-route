FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl git dos2unix && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir uvicorn fastapi streamlit -r requirements.txt
RUN dos2unix start.sh && chmod +x start.sh
EXPOSE 7860
CMD ["./start.sh"]