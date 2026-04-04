from src.environment import TrafficEnv
import time

# 1. Start the world
env = TrafficEnv()
# Reset returns the first observation
obs = env.reset()

print("--- Starting Phase 2 Test Run ---")
print(f"Initial State: {obs}")

# 2. Run a mini-simulation (5 steps)
# We will alternate the light to see if cars move and stop
for i in range(1, 6):
    # Action 0: NS_GREEN, Action 1: EW_GREEN
    # Let's keep it North-South Green (0) so the ambulance moves
    action = 0 
    
    # The step function now returns 3 things!
    obs, reward, done = env.step(action)
    
    print(f"Step {i}:")
    print(f"  Action Taken: {'NS_GREEN' if action == 0 else 'EW_GREEN'}")
    print(f"  Ambulance Pos: {obs[0].item()}, {obs[1].item()}")
    print(f"  Current Reward: {reward}")
    print(f"  Is Mission Done?: {done}")
    print("-" * 20)
    
    if done:
        print("Goal Reached! The ambulance has cleared the intersection.")
        break
    
    time.sleep(0.5) # Slow down the output so you can read it