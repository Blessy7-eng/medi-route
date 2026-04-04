# 1. Use the official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system-level dependencies (needed for RL libraries)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 4. Copy everything from your current folder into the container
COPY . /app

# 5. Install the Python libraries
# We added 'openai' because your inference.py now needs it for the mandatory LLM call
RUN pip install --no-cache-dir torch stable-baselines3 shimmy gymnasium pandas openai

# 6. Tell Docker to run your inference script when it starts
# Replace your current CMD with this:
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]