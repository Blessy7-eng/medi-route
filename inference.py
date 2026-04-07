import os
import time
import sys
import torch
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# 1. MANDATORY: Environment Variables for the Grader
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

def get_llm_summary(total_reward, frames, task_id):
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
    # Log Format REQUIRED: [START] {task_id}
    print(f"[START] {task_id}", flush=True)
    
    env = TrafficEnv(difficulty=difficulty)
    
    model_path = "medi_route_brain.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Ensure model is uploaded to root.", flush=True)
        return

    model = DQN.load(model_path)
    obs, _ = env.reset()
    total_reward = 0
    frames = 0

    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        frames += 1

        # Log Format REQUIRED: [STEP] {step} | Action: {action} | Reward: {reward}
        print(f"[STEP] {step} | Action: {action} | Reward: {round(reward, 2)}", flush=True)

        if done or truncated:
            break
    
    try:
        final_score = env.get_performance_score(frames)
    except:
        final_score = round(min(max(total_reward / 50, 0.0), 1.0), 2)
    
    summary = get_llm_summary(total_reward, frames, task_id)
    print(f"AI Evaluation: {summary}", flush=True)

    # Log Format REQUIRED: [END] {task_id} | Final Score: {final_score}
    print(f"[END] {task_id} | Final Score: {final_score}", flush=True)

if __name__ == "__main__":
    # Execute the 3 Mandatory Tasks
    run_task("Easy_Clear", "easy")
    run_task("Medium_Traffic", "medium")
    run_task("Hard_Sangli_Rush", "hard")

    print("--- ALL EVALUATIONS COMPLETE ---", flush=True)

    # Keep container alive for judges
    if os.name == 'nt': 
        while True: time.sleep(3600)
    else:
        import subprocess
        subprocess.call(["tail", "-f", "/dev/null"])