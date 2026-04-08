import os
import sys
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
                  f"{round(total_reward, 2)} in {frames} steps. Provide a 1-sentence evaluation.")
        completion = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=60
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "LLM Summary Status: Connected but returned error."

def run_grader_evaluation():
    scenarios = [("Easy_Clear", "easy"), ("Medium_Traffic", "medium"), ("Hard_Sangli_Rush", "hard")]
    model_path = "medi_route_brain.zip"
    
    try:
        model = DQN.load(model_path)
    except Exception:
        model = None
        print(f"⚠️ Warning: {model_path} not found.")

    for task_id, diff in scenarios:
        print(f"[START] {task_id}", flush=True)
        env = TrafficEnv(difficulty=diff)
        obs, _ = env.reset()
        total_reward, frames = 0, 0

        for step in range(100):
            action = int(model.predict(obs, deterministic=True)[0]) if model else 0
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            frames += 1
            
            # Grader needs to see these specific tags
            print(f"[STEP] {step} | Action: {action} | Reward: {round(reward, 2)}", flush=True)
            
            if done or truncated: 
                break
        
        summary = get_llm_summary(total_reward, frames, task_id)
        final_score = round(min(max(total_reward / 50, 0.0), 1.0), 2)
        print(f"AI Evaluation: {summary}", flush=True)
        print(f"[END] {task_id} | Final Score: {final_score}", flush=True)

if __name__ == "__main__":
    # ONLY run logic here. Do NOT start uvicorn/FastAPI servers.
    run_grader_evaluation()
    print("--- ALL EVALUATIONS COMPLETE ---", flush=True)