from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from src.environment import TrafficEnv

# Initialize the API and Environment
app = FastAPI()
env_api = TrafficEnv(difficulty="medium")

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any]

@app.post("/reset")
def reset_endpoint():
    obs, info = env_api.reset()
    # Ensure we return a standard dictionary with list-converted observation
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
def step_endpoint(action: int):
    obs, reward, done, truncated, info = env_api.step(action)
    
    # ADD THIS LINE to see it in your terminal:
    print(f"[STEP] Action: {action}, Reward: {reward:.2f}")
    
    if done or truncated:
        print(f"[END] Final Score: {reward:.2f}")

    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(done or truncated),
        "info": info
    }

if __name__ == "__main__":
    # Port 7860 is mandatory for Hugging Face Spaces
    print("🚀 Medi-Route API Server starting on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)