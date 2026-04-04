from stable_baselines3 import DQN
from src.environment import TrafficEnv
import os

# 1. Initialize our world with Medium difficulty
# This gives the AI enough traffic to learn but not so much that it gets overwhelmed
env = TrafficEnv(difficulty="medium")

# 2. Create the Brain (DQN)
# 'MlpPolicy' = Multi-layer Perceptron (the standard AI brain)
# verbose=1 = Show us the progress in the terminal
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    exploration_fraction=0.2  # AI spends 20% of time "exploring" new moves
)

# 3. Train for 10,000 steps
# This usually takes 1-3 minutes depending on your computer
print("🚀 Training started... The AI is practicing the Sangli routes!")
model.learn(total_timesteps=10000)

# 4. Save the brain
model.save("medi_route_brain")
print("✅ Training complete! Brain saved as 'medi_route_brain.zip'")