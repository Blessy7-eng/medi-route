import streamlit as st
import time
import torch
import os
import sys
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from threading import Thread
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# --- 1. OpenEnv API Compliance (MANDATORY FOR GRADER) ---
# This section ensures the ./validate-submission.sh script passes Step 1
api = FastAPI()
env_api = TrafficEnv(difficulty="medium")

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any]

@api.post("/reset")
def reset_endpoint():
    obs, info = env_api.reset()
    return {"observation": obs.tolist(), "info": info}

@api.post("/step")
def step_endpoint(action: int):
    obs, reward, done, truncated, info = env_api.step(action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(done or truncated),
        "info": info
    }

# Function to run the API in the background
def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000)

# Start API thread if not already running
if 'api_started' not in st.session_state:
    thread = Thread(target=run_api, daemon=True)
    thread.start()
    st.session_state.api_started = True

# --- 2. Hackathon Compliance: Mandatory Grader Logs ---
if 'grader_run' not in st.session_state:
    # This triggers your inference.py to generate [START], [STEP], [END] tags
    subprocess.Popen([sys.executable, "inference.py"])
    st.session_state.grader_run = True

# --- 3. Streamlit UI (The "Sangli" Visualizer) ---
st.set_page_config(page_title="Medi-Route Sangli", layout="wide", page_icon="🚑")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

if 'env' not in st.session_state:
    st.session_state.env = TrafficEnv(difficulty="medium")
    st.session_state.total_reward = 0
    st.session_state.done = False

env = st.session_state.env

@st.cache_resource
def load_brain():
    if os.path.exists("medi_route_brain.zip"):
        return DQN.load("medi_route_brain.zip")
    return None

model = load_brain()

# --- 4. Sidebar Controls ---
st.sidebar.header("🕹️ Simulation Control")
st.sidebar.subheader("📉 Live Progress")
current_steps = getattr(env, 'steps', 0)
st.sidebar.progress(min(current_steps / 50, 1.0)) 
st.sidebar.write(f"⏱️ Step: {current_steps} / 50")

difficulty = st.sidebar.selectbox("Select Scenario", ["easy", "medium", "hard"])
mode = st.sidebar.radio("Control Mode", ["AI Optimized", "Manual Override"])

if st.sidebar.button("🚀 Restart Simulation"):
    st.session_state.env = TrafficEnv(difficulty=difficulty)
    st.session_state.total_reward = 0
    st.session_state.done = False
    st.rerun()

st.sidebar.header("📊 Real-Time Analytics")
score_metric = st.sidebar.empty()
speed_metric = st.sidebar.empty()
traffic_metric = st.sidebar.empty()

# --- 5. Main Simulation UI ---
st.title("🚑 Medi-Route: AI Emergency Response")
st.subheader("📍 Location: Sangli-Miraj Road")

grid_placeholder = st.empty()
signal_col1, signal_col2 = st.columns(2)

if not st.session_state.done:
    for frame in range(100):
        # AI or Manual Decision
        if mode == "AI Optimized" and model is not None:
            obs = env.get_observation()
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = 0 # Default to NS_GREEN for demo flow

        obs, reward, done, _, _ = env.step(action)
        st.session_state.total_reward += reward

        # UI Updates
        with signal_col1: st.markdown(f"### NS Signal: {'🟢' if action == 0 else '🔴'}")
        with signal_col2: st.markdown(f"### EW Signal: {'🟢' if action == 1 else '🔴'}")

        # Grid Rendering
        grid = [["⬛" for _ in range(10)] for _ in range(10)]
        for i in range(10): grid[5][i] = "🛣️"; grid[i][5] = "🛣️"
        for v in env.vehicles:
            icon = "🚑" if v.get('ev') else "🚗"
            grid[min(max(int(v['y']), 0), 9)][min(max(int(v['x']), 0), 9)] = icon
        
        grid_placeholder.table(grid)
        
        # Metrics
        score_metric.metric("Total Reward", f"{st.session_state.total_reward:.1f}")
        speed_metric.metric("Ambulance Speed", "60 km/h" if any(v.get('ev') and v['speed'] > 0 for v in env.vehicles) else "0 km/h")

        if done:
            st.session_state.done = True
            st.success("Ambulance Reached Hospital! 🎉")
            break
        time.sleep(0.3)