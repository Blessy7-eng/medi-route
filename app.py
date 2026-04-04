import streamlit as st
import time
import torch
from src.environment import TrafficEnv

# Page Config
st.set_page_config(page_title="Medi-Route Sim", layout="wide")
st.title("🚑 Medi-Route: AI Emergency Dispatch")

# Sidebar for stats
st.sidebar.header("Live Simulation Metrics")
score_placeholder = st.sidebar.empty()
light_placeholder = st.sidebar.empty()
obs_placeholder = st.sidebar.empty()

# Initialize Environment
if 'env' not in st.session_state:
    st.session_state.env = TrafficEnv()

env = st.session_state.env
grid_placeholder = st.empty()

# Reset button
if st.sidebar.button("Restart Simulation"):
    env.reset()
    st.rerun()

# Simulation Loop
for frame in range(50):
    # 1. Choose Action (For now, let's alternate every 5 steps to test)
    # 0 = NS_GREEN, 1 = EW_GREEN
    action = 0 if (frame // 5) % 2 == 0 else 1
    
    # 2. Step the environment
    obs, reward, done = env.step(action)
    
    # 3. Create the Visual Grid
    # We use a 10x10 grid of emojis
    # ... (inside your for frame in range(50) loop) ...

    grid = [["⬛" for _ in range(10)] for _ in range(10)]

    # Draw the 4-way intersection roads
    for i in range(10):
        grid[5][i] = "🛣️" # East-West Road
        grid[i][5] = "🛣️" # North-South Road

    # Draw Vehicles
    for v in env.vehicles:
        icon = "🚑" if v.is_emergency else "🚗"
        grid[int(v.y)][int(v.x)] = icon

    grid_placeholder.table(grid)
    
    # Update Sidebar Stats
    current_light = "🟢 GREEN (NS)" if action == 0 else "🔴 RED (NS)"
    light_placeholder.metric("Traffic Light", current_light)
    score_placeholder.metric("Current Reward", f"{reward:.2f}")
    obs_placeholder.text(f"Observation: {obs.tolist()}")

    if done:
        st.success("Ambulance Reached Destination! 🎉")
        break

    time.sleep(0.4) # Control speed of simulation