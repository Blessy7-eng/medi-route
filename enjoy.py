import time
import streamlit as st
from stable_baselines3 import DQN
from src.environment import TrafficEnv

# 1. Load the Environment and the Trained Brain
env = TrafficEnv(difficulty="hard")
model = DQN.load("medi_route_brain")

def run_ai_demo():
    st.title("🚑 AI in Control: Medi-Route Demo")
    grid_placeholder = st.empty()
    stats_placeholder = st.sidebar.empty()
    
    obs, info = env.reset()
    total_reward = 0
    
    # Run for 100 "ticks" of the clock
    for step in range(100):
        # The AI chooses the best action (0 or 1)
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply the action to the world
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # --- Member 3's Visualizer Logic ---
        # Create the emoji grid
        grid = [["⬛" for _ in range(10)] for _ in range(10)]
        for i in range(10):
            grid[5][i] = "🛣️" # EW Road
            grid[i][5] = "🛣️" # NS Road
            
        # Draw Vehicles
        for v in env.vehicles:
            icon = "🚑" if v['ev'] else "🚗"
            
            # 🛑 FIX: Clip coordinates to stay inside the 10x10 grid (0-9)
            grid_y = max(0, min(9, int(v['y'])))
            grid_x = max(0, min(9, int(v['x'])))
            
            grid[grid_y][grid_x] = icon
            
        # Update the UI
        grid_placeholder.table(grid)
        stats_placeholder.write(f"**Step:** {step}")
        stats_placeholder.write(f"**Action:** {'Green NS' if action == 0 else 'Green EW'}")
        stats_placeholder.write(f"**Reward:** {reward:.2f}")
        
        if done:
            st.success(f"Ambulance arrived! Total Reward: {total_reward:.2f}")
            break
            
        time.sleep(0.3) # Slow it down so humans can see the magic

if __name__ == "__main__":
    run_ai_demo()