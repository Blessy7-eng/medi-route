FROM python:3.10

WORKDIR /app

# Copy everything to ensure inference.py is at the root
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "inference.py"]