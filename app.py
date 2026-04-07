import streamlit as st
import time
import os
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# --- 1. Wrap everything in a main() function for OpenEnv ---
def main():
    # --- UI Configuration ---
    st.set_page_config(page_title="Medi-Route Sangli", layout="wide", page_icon="🚑")

    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
        </style>
        """, unsafe_allow_html=True)

    # Initialize Session State
    if 'env' not in st.session_state:
        st.session_state.env = TrafficEnv(difficulty="medium")
        st.session_state.total_reward = 0
        st.session_state.done = False

    env = st.session_state.env

    @st.cache_resource
    def load_brain():
        if os.path.exists("medi_route_brain.zip"):
            try:
                return DQN.load("medi_route_brain.zip")
            except:
                return None
        return None

    model = load_brain()

    # --- Sidebar ---
    st.sidebar.header("🕹️ Simulation Control")
    difficulty = st.sidebar.selectbox("Select Scenario", ["easy", "medium", "hard"])
    mode = st.sidebar.radio("Control Mode", ["AI Optimized", "Manual Override"])

    if st.sidebar.button("🚀 Restart Simulation"):
        st.session_state.env = TrafficEnv(difficulty=difficulty)
        st.session_state.total_reward = 0
        st.session_state.done = False
        st.rerun()

    st.sidebar.header("📊 Real-Time Analytics")
    score_metric = st.sidebar.empty()

    # --- Main UI ---
    st.title("🚑 Medi-Route: AI Emergency Response")
    st.subheader("📍 Location: Sangli-Miraj Road")

    grid_placeholder = st.empty()
    signal_col1, signal_col2 = st.columns(2)

    # Use a container for the status so it stays at the top
    status_msg = st.empty()

    # --- Simulation Loop ---
    if not st.session_state.done:
        # We use a placeholder to prevent UI flickering
        while not st.session_state.done:
            # Decision Logic
            if mode == "AI Optimized" and model is not None:
                # Use the environment's observation method
                obs, _ = env.reset() if st.session_state.total_reward == 0 else (env._get_obs(), {})
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = 0 # Default manual flow

            obs, reward, done, truncated, info = env.step(int(action))
            st.session_state.total_reward += reward

            # Signal UI
            signal_col1.markdown(f"### NS Signal: {'🟢' if action == 0 else '🔴'}")
            signal_col2.markdown(f"### EW Signal: {'🟢' if action == 1 else '🔴'}")

            # Grid Rendering (10x10)
            grid = [["⬛" for _ in range(10)] for _ in range(10)]
            for i in range(10): 
                grid[5][i] = "🛣️" # Horizontal Road
                grid[i][5] = "🛣️" # Vertical Road
                
            for v in env.vehicles:
                icon = "🚑" if v.get('ev') else "🚗"
                # Boundary safety for 10x10 grid
                py = min(max(int(v['y']), 0), 9)
                px = min(max(int(v['x']), 0), 9)
                grid[py][px] = icon
            
            grid_placeholder.table(grid)
            
            # Update Analytics
            score_metric.metric("Total Reward", f"{st.session_state.total_reward:.1f}")
            
            if done or truncated:
                st.session_state.done = True
                status_msg.success("Ambulance Reached Hospital! 🎉")
                st.balloons()
                break
            
            time.sleep(0.3)

if __name__ == "__main__":
    main()