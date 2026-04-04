import streamlit as st
import time
import torch
import os
import sys
import subprocess
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# --- 0. Hackathon Compliance: Mandatory Grader Logs ---
if 'grader_run' not in st.session_state:
    # Use sys.executable to ensure we use the same python path as Streamlit
    # stdout=None and stderr=None allows it to print directly to the container console
    subprocess.Popen(
        [sys.executable, "inference.py"], 
        stdout=None, 
        stderr=None, 
        bufsize=1, 
        universal_newlines=True
    )
    st.session_state.grader_run = True

# --- 1. Page Config & Branding ---
st.set_page_config(page_title="Medi-Route Sangli", layout="wide", page_icon="🚑")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚑 Medi-Route: AI Emergency Response")
st.subheader("📍 Location: Sangli-Miraj Road (High Density Corridor)")
st.caption("Simulating North-South Flow at Ganpati Mandir Road Intersection")

# --- 2. Sidebar: Controls & Live Analytics ---
st.sidebar.header("🕹️ Simulation Control")
st.sidebar.subheader("📉 Live Progress")
st.sidebar.progress(min(env.steps / 50, 1.0)) # Shows a progress bar toward timeout
st.sidebar.write(f"⏱️ Step: {env.steps} / 50")
difficulty = st.sidebar.selectbox("Select Scenario", ["easy", "medium", "hard"])
mode = st.sidebar.radio("Control Mode", ["AI Optimized", "Manual Override"])

manual_action = 0
if mode == "Manual Override":
    is_ew_green = st.sidebar.toggle("Switch to EW Green", value=False)
    manual_action = 1 if is_ew_green else 0

if st.sidebar.button("🚀 Restart Simulation"):
    st.session_state.env = TrafficEnv(difficulty=difficulty)
    st.session_state.total_reward = 0
    st.session_state.done = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📊 Real-Time Analytics")
score_metric = st.sidebar.empty()
speed_metric = st.sidebar.empty()
traffic_metric = st.sidebar.empty()

# --- 3. Initialize Environment & Model ---
if 'env' not in st.session_state:
    st.session_state.env = TrafficEnv(difficulty="medium")
    st.session_state.total_reward = 0
    st.session_state.done = False

# Load the AI Brain once
@st.cache_resource
def load_brain():
    if os.path.exists("medi_route_brain.zip"):
        return DQN.load("medi_route_brain.zip")
    return None

model = load_brain()
env = st.session_state.env
grid_placeholder = st.empty()
signal_col1, signal_col2 = st.columns(2)

# --- 4. Simulation Loop ---
if not st.session_state.done:
    # Small notice to the judge that logs are being generated
    st.toast("Grader script running in background... Check 'Container Logs' for scoring tags.")
    
    for frame in range(100):
        # 1. Decision Logic
        if mode == "AI Optimized" and model is not None:
            obs = env.get_observation()
            action, _ = model.predict(obs, deterministic=True)
        elif mode == "AI Optimized":
            # Heuristic fallback if model isn't trained yet
            action = 0 if any(v.get('ev') and v['dir'] == "NS" for v in env.vehicles) else 1
        else:
            action = manual_action

        # 2. Step the environment
        obs, reward, done, _, _ = env.step(action)
        st.session_state.total_reward += reward

        # 3. UI: Signal Phase Indicators
        with signal_col1:
            light_icon = "🟢" if action == 0 else "🔴"
            st.markdown(f"### NS Signal: {light_icon}")
        with signal_col2:
            light_icon = "🟢" if action == 1 else "🔴"
            st.markdown(f"### EW Signal: {light_icon}")

        # 4. Create the Visual Grid
        grid = [["⬛" for _ in range(10)] for _ in range(10)]
        for i in range(10):
            grid[5][i] = "🛣️" # East-West
            grid[i][5] = "🛣️" # North-South

        for v in env.vehicles:
            icon = "🚑" if v.get('ev') and v['speed'] > 0 else "🚨" if v.get('ev') else "🚗" if v['speed'] > 0 else "🛑"
            y_pos = min(max(int(v['y']), 0), 9)
            x_pos = min(max(int(v['x']), 0), 9)
            grid[y_pos][x_pos] = icon

        grid_placeholder.table(grid)

        # 5. Update Stat Counters
        amb_speed = 60 if any(v.get('ev') and v['speed'] > 0 for v in env.vehicles) else 0
        waiting_cars = len([v for v in env.vehicles if v['speed'] == 0 and not v.get('ev')])
        
        score_metric.metric("Total Reward", f"{st.session_state.total_reward:.1f}", delta=f"{reward:.1f}")
        speed_metric.metric("Ambulance Speed", f"{amb_speed} km/h")
        traffic_metric.metric("Waiting Traffic", f"{waiting_cars} Vehicles")

        if done:
            st.session_state.done = True
            st.balloons()
            st.success(f"Ambulance Reached Hospital! Total Steps: {frame} 🎉")
            break

        time.sleep(0.3)