import os
import time
import torch
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# 1. MANDATORY: Environment Variables for the Grader
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
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

if __name__ == "__main__":
    run_task("Easy_Clear", "easy")
    run_task("Medium_Traffic", "medium")
    run_task("Hard_Sangli_Rush", "hard")

    # CRITICAL: Keep the container alive for the judges
    print("--- ALL TASKS COMPLETE ---")
    import subprocess
    # This keeps the container awake without using CPU
    subprocess.call(["tail", "-f", "/dev/null"])
