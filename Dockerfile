FROM python:3.10

WORKDIR /app

# Copy everything from your local folder to the container
COPY . .

RUN pip install -r requirements.txt

# Make start.sh executable
RUN chmod +x start.sh

# The port Hugging Face expects
EXPOSE 7860

CMD ["./start.sh"]