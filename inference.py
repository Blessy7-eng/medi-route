import os
import sys
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# --- 1. Setup API and Environment ---
app = FastAPI()

# CRITICAL: Allow the Scaler Validator to send POST requests (Fixes 403 Forbidden)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env_api = TrafficEnv(difficulty="medium")

# MANDATORY: Environment Variables for the Grader
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

# --- 2. LLM Summary Logic ---
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
        return f"LLM Summary Status: Error: {str(e)}"

# --- 3. Grader Evaluation Loop (Logs for Phase 2/3) ---
def run_grader_evaluation():
    scenarios = [("Easy_Clear", "easy"), ("Medium_Traffic", "medium"), ("Hard_Sangli_Rush", "hard")]
    model_path = "medi_route_brain.zip"
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DQN.load(model_path, device=device)
    except Exception as e:
        print(f"❌ Error loading model: {e}", flush=True)
        return

    for task_id, diff in scenarios:
        print(f"[START] {task_id}", flush=True)
        env = TrafficEnv(difficulty=diff)
        obs, _ = env.reset()
        total_reward, frames = 0, 0

        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            total_reward += reward
            frames += 1
            
            # MANDATORY LOG FORMAT
            print(f"[STEP] {step} | Action: {action} | Reward: {round(reward, 2)}", flush=True)
            if done or truncated: break
        
        summary = get_llm_summary(total_reward, frames, task_id)
        final_score = round(min(max(total_reward / 100, 0.0), 1.0), 2)
        print(f"AI Evaluation: {summary}", flush=True)
        print(f"[END] {task_id} | Final Score: {final_score}", flush=True)

# --- 4. API Endpoints (Required for Scaler Ping) ---
@app.get("/")
def read_root():
    return {"status": "Medi-Route API is running", "location": "Sangli-Miraj Road"}

@app.post("/reset")
def reset_endpoint():
    obs, info = env_api.reset()
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
def step_endpoint(action: int):
    obs, reward, done, truncated, info = env_api.step(action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(done or truncated),
        "info": info
    }

# --- 5. Execution Entry Point ---
if __name__ == "__main__":
    # FIRST: Run the offline evaluation loop for the grader logs
    print("🚀 PHASE 1: Running Grader Evaluation Loop...", flush=True)
    run_grader_evaluation()
    
    # SECOND: Start the server to pass the Phase 2 "Ping" check
    print("📡 PHASE 2: Starting API Server on Port 7860...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=7860)