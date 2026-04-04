import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TrafficEnv(gym.Env):
    # Inside your TrafficEnv class
    @property
    def ambulance_speed_kmh(self):
        # Convert your grid speed (1 or 2) to a "realistic" km/h for the UI
        amb = [v for v in self.vehicles if v.get('ev')]
        if not amb: return 0
        return 60 if amb[0]['speed'] > 0 else 0

    @property
    def waiting_cars_count(self):
        return len([v for v in self.vehicles if v['speed'] == 0 and not v.get('ev')])
    
    def __init__(self, difficulty="medium"):
        super(TrafficEnv, self).__init__()
        self.grid_size = 10
        self.difficulty = difficulty  # "easy", "medium", or "hard"
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 20, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicles = []
        self.light_state = "NS_GREEN"
        
        # --- Step 2: Implementing the "Lessons" (Scenarios) ---
        # 1. Always spawn the Emergency Vehicle
        self.spawn_vehicle(direction="NS", is_emergency=True)
        
        # 2. Spawn normal cars based on difficulty
        if self.difficulty == "easy":
            pass # 0 normal cars
        elif self.difficulty == "medium":
            for _ in range(5):
                self.spawn_vehicle(direction=random.choice(["NS", "EW"]), is_emergency=False)
        elif self.difficulty == "hard":
            for _ in range(10): # Start with 10
                self.spawn_vehicle(direction=random.choice(["NS", "EW"]), is_emergency=False)
        
        return self.get_observation(), {}

    def spawn_vehicle(self, direction="NS", is_emergency=False):
        if direction == "NS":
            v = {'x': 5, 'y': 0, 'dir': "NS", 'ev': is_emergency, 'speed': 0}
        else:
            v = {'x': 0, 'y': 5, 'dir': "EW", 'ev': is_emergency, 'speed': 0}
        self.vehicles.append(v)

    def step(self, action):
        self.light_state = "NS_GREEN" if action == 0 else "EW_GREEN"
        
        reward = 0
        for v in self.vehicles:
            is_green = (v['dir'] == "NS" and self.light_state == "NS_GREEN") or \
                       (v['dir'] == "EW" and self.light_state == "EW_GREEN")
            
            # Stop line logic at index 4
            at_stop = (v['dir'] == "NS" and v['y'] == 4) or (v['dir'] == "EW" and v['x'] == 4)
            
            if is_green or not at_stop:
                v['speed'] = 2 if v['ev'] else 1
            else:
                v['speed'] = 0
                
            if v['dir'] == "NS": v['y'] += v['speed']
            else: v['x'] += v['speed']

            # Reward logic
            if v['ev']:
                reward += 2.0 if v['speed'] > 0 else -5.0
            elif v['speed'] == 0:
                reward -= 0.1

        # --- Hard Scenario: Continuous Spawning (Sangli Market) ---
        if self.difficulty == "hard":
            # 30% chance to spawn a new car every single step
            if random.random() < 0.3:
                self.spawn_vehicle(direction=random.choice(["NS", "EW"]), is_emergency=False)

        obs = self.get_observation()
        # Done if ambulance reaches goal OR if 100 steps have passed
        self.steps = getattr(self, 'steps', 0) + 1
        reached_goal = any(v['ev'] and (v['x'] >= 9 or v['y'] >= 9) for v in self.vehicles)
        done = reached_goal or self.steps >= 100
        # Add this inside step() in src/environment.py before returning
        self.vehicles = [v for v in self.vehicles if v['x'] < 10 and v['y'] < 10]
        return obs, reward, done, False, {}

    def get_observation(self):
        ev_list = [v for v in self.vehicles if v['ev']]
        ev = ev_list[0] if ev_list else {'x': 0, 'y': 0}
        waiting = len([v for v in self.vehicles if v['speed'] == 0 and not v['ev']])
        light = 0 if self.light_state == "NS_GREEN" else 1
        return np.array([ev['x'], ev['y'], waiting, light], dtype=np.float32)
    
    def get_performance_score(self, frames_taken):
        """
        Translates simulation time into a 0.0 - 1.0 score for the judges.
        - Perfect (15 frames): 1.0
        - Slow (100 frames): 0.2
        - Failed: 0.0
        """
        if frames_taken <= 15:
            return 1.0
        elif frames_taken >= 100:
            return 0.0
        else:
            # Scale linearly: fewer frames = higher score
            score = 1.0 - ((frames_taken - 15) / (100 - 15))
            return round(max(0.2, score), 2)