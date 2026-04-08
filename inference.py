import os
import sys
import torch
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# MANDATORY: Environment Variables for the Grader
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

def get_llm_summary(total_reward, frames, task_id):
    if not API_KEY:
        return "Grader Note: LLM Summary skipped (HF_TOKEN missing in Secrets)"
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        prompt = (f"In a traffic simulation for {task_id}, the AI achieved a reward of "
                  f"{round(total_reward, 2)} in {frames} steps. Provide a 1-sentence technical evaluation.")
        
        completion = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=60
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Summary Status: Error calling model: {str(e)}"

def run_grader_evaluation():
    # The 3 Mandatory Tasks
    scenarios = [("Easy_Clear", "easy"), ("Medium_Traffic", "medium"), ("Hard_Sangli_Rush", "hard")]
    model_path = "medi_route_brain.zip"
    
    # Load the trained model
    try:
        if os.path.exists(model_path):
            model = DQN.load(model_path)
        else:
            print(f"⚠️ Warning: {model_path} not found. Ensure it is in the root directory.", flush=True)
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}", flush=True)
        return

    for task_id, diff in scenarios:
        # LOG FORMAT REQUIRED: [START] {task_id}
        print(f"[START] {task_id}", flush=True)
        
        env = TrafficEnv(difficulty=diff)
        obs, _ = env.reset()
        total_reward, frames = 0, 0

        # Simulation loop (max 100 steps)
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            
            total_reward += reward
            frames += 1
            
            # LOG FORMAT REQUIRED: [STEP] {step} | Action: {action} | Reward: {reward}
            print(f"[STEP] {step} | Action: {action} | Reward: {round(reward, 2)}", flush=True)
            
            if done or truncated: 
                break
        
        # Calculate performance and get LLM summary
        summary = get_llm_summary(total_reward, frames, task_id)
        
        try:
            final_score = env.get_performance_score(frames)
        except:
            final_score = round(min(max(total_reward / 100, 0.0), 1.0), 2)

        print(f"AI Evaluation: {summary}", flush=True)
        # LOG FORMAT REQUIRED: [END] {task_id} | Final Score: {final_score}
        print(f"[END] {task_id} | Final Score: {final_score}", flush=True)

if __name__ == "__main__":
    # CRITICAL: This script must ONLY print to stdout and exit.
    # It must NOT start any FastAPI, Uvicorn, or Streamlit servers.
    run_grader_evaluation()
    print("--- ALL EVALUATIONS COMPLETE ---", flush=True)
    sys.exit(0)