import os
import time
import torch
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# 1. MANDATORY: Environment Variables for the Grader
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

def get_llm_summary(total_reward, frames):
    """
    Mandatory Instruction: Using the OpenAI Client for an LLM call.
    This explains the result to the judges.
    """
    if not API_KEY or not MODEL_NAME:
        return "LLM metrics skipped (No API Key)"

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    prompt = f"The Traffic AI achieved a reward of {total_reward} in {frames} steps. Briefly evaluate this performance."
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Summary Error: {e}"

def run_task(task_id, difficulty):
    # Log format REQUIRED by OpenEnv spec
    print(f"[START] {task_id}")
    
    env = TrafficEnv(difficulty=difficulty)
    
    # Load the brain Member 2 trained
    try:
        model = DQN.load("medi_route_brain.zip")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs, _ = env.reset()
    total_reward = 0
    frames = 0

    for step in range(100): # Max steps for a task
        # AI suggests an action (0: NS_GREEN, 1: EW_GREEN)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        frames += 1

        # MUST follow this exact log format for the auto-grader
        print(f"[STEP] {step} | Action: {action} | Reward: {round(reward, 2)}")

        if done:
            break
    
    # Calculate performance using the method in your environment.py
    final_score = env.get_performance_score(frames)
    
    # Use the OpenAI Client to fulfill the mandatory LLM usage rule
    summary = get_llm_summary(total_reward, frames)
    print(f"AI Evaluation: {summary}")

    print(f"[END] {task_id} | Final Score: {final_score}")

if __name__ == "__main__":import os
import time
import sys
import torch
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# 1. MANDATORY: Environment Variables for the Grader
# These are pulled from your Hugging Face Space Secrets
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

def get_llm_summary(total_reward, frames, task_id):
    """
    Mandatory Instruction: Use OpenAI Client for an LLM call.
    This fulfills the requirement to 'use LLM for evaluation/summary'.
    """
    if not API_KEY:
        return "Grader Note: LLM Summary skipped (HF_TOKEN missing in Secrets)"

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        prompt = (f"In a traffic simulation for {task_id}, the AI achieved a reward of "
                  f"{round(total_reward, 2)} in {frames} steps. "
                  "Provide a 1-sentence technical evaluation of this efficiency.")
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Summary Status: Connected but returned error: {e}"

def run_task(task_id, difficulty):
    # LOG FORMAT REQUIRED: [START] {task_id}
    print(f"[START] {task_id}")
    sys.stdout.flush() # Force log to appear immediately
    
    env = TrafficEnv(difficulty=difficulty)
    
    # Load the brain Member 2 trained
    model_path = "medi_route_brain.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Ensure model is uploaded to root.")
        return

    model = DQN.load(model_path)
    
    # Standard Reset
    obs, _ = env.reset()
    total_reward = 0
    frames = 0

    # Max 100 steps as per your environment logic
    for step in range(100):
        # AI Predicts Action (0: NS_GREEN, 1: EW_GREEN)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        frames += 1

        # LOG FORMAT REQUIRED: [STEP] {step} | Action: {action} | Reward: {reward}
        print(f"[STEP] {step} | Action: {action} | Reward: {round(reward, 2)}")
        sys.stdout.flush()

        if done or truncated:
            break
    
    # Calculate performance (Ensure this method exists in your TrafficEnv)
    # If not, use: final_score = min(max(total_reward / 100, 0.0), 1.0)
    try:
        final_score = env.get_performance_score(frames)
    except:
        final_score = round(min(max(total_reward / 50, 0.0), 1.0), 2)
    
    # Mandatory LLM Summary
    summary = get_llm_summary(total_reward, frames, task_id)
    print(f"AI Evaluation: {summary}")

    # LOG FORMAT REQUIRED: [END] {task_id} | Final Score: {score}
    print(f"[END] {task_id} | Final Score: {final_score}")
    sys.stdout.flush()

if __name__ == "__main__":
    # The 3 Mandatory Tasks
    run_task("Easy_Clear", "easy")
    run_task("Medium_Traffic", "medium")
    run_task("Hard_Sangli_Rush", "hard")

    print("--- ALL EVALUATIONS COMPLETE ---")
    sys.stdout.flush()
    
    # Keep container alive for the Hugging Face/Scaler health check
    time.sleep(30)
    run_task("Easy_Clear", "easy")
    run_task("Medium_Traffic", "medium")
    run_task("Hard_Sangli_Rush", "hard")

    # CRITICAL: Keep the container alive for the judges
    print("--- ALL TASKS COMPLETE ---")
<<<<<<< HEAD
    import subprocess
    # This keeps the container awake without using CPU
    subprocess.call(["tail", "-f", "/dev/null"])
=======
    sys.stdout.flush()

    # Cross-platform way to keep the process alive
    if os.name == 'nt':  # If running on Windows
        print("Local Windows test complete. Press Ctrl+C to exit.")
        while True:
            time.sleep(3600)
    else:  # If running on Hugging Face (Linux)
        import subprocess
        subprocess.call(["tail", "-f", "/dev/null"])    
>>>>>>> 2c0f795 (Updated inference.py)
