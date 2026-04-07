import os
import time
import sys
import torch
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# 1. MANDATORY: Environment Variables for the Grader
# Use the Router URL for better stability during the hackathon
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
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
    sys.stdout.flush() 
    
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
    
    # Calculate performance
    try:
        final_score = env.get_performance_score(frames)
    except:
        final_score = round(min(max(total_reward / 50, 0.0), 1.0), 2)
    
    # Mandatory LLM Summary
    summary = get_llm_summary(total_reward, frames, task_id)
    print(f"AI Evaluation: {summary}")

    # LOG FORMAT REQUIRED: [END] {task_id} | Final Score: {final_score}
    print(f"[END] {task_id} | Final Score: {final_score}")
    sys.stdout.flush()

if __name__ == "__main__":
    # The 3 Mandatory Tasks
    run_task("Easy_Clear", "easy")
    run_task("Medium_Traffic", "medium")
    run_task("Hard_Sangli_Rush", "hard")

    print("--- ALL EVALUATIONS COMPLETE ---")
    sys.stdout.flush()

    # Keep the container alive for the judges/health checks
    if os.name == 'nt':  # Windows local test
        print("Local Windows test complete. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:  # Hugging Face (Linux)
        import subprocess
        subprocess.call(["tail", "-f", "/dev/null"])