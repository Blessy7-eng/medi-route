import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def Vehicle(x, y, is_emergency=False, direction="NS"):
    """
    Creates a vehicle dictionary compatible with your current logic.
    """
    return {
        'x': x,
        'y': y,
        'ev': is_emergency,
        'dir': direction,
        'speed': 0
    }

class TrafficEnv(gym.Env):
    def __init__(self, difficulty="medium"):
        super(TrafficEnv, self).__init__()
        
        # 1. IMMEDIATE INITIALIZATION to prevent NameErrors/AttributeErrors
        self.grid_size = 10
        self.difficulty = difficulty
        self.vehicles = [] 
        self.steps = 0
        self.light_state = "NS_GREEN"
        
        # 2. Define Action and Observation Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 20, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # 3. Initial Reset
        self.reset()

    @property
    def ambulance_speed_kmh(self):
        amb = [v for v in self.vehicles if v.get('ev')]
        if not amb: return 0
        return 60 if amb[0]['speed'] > 0 else 0

    @property
    def waiting_cars_count(self):
        return len([v for v in self.vehicles if v['speed'] == 0 and not v.get('ev')])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicles = [] 
        self.steps = 0     
        self.light_state = "NS_GREEN" 
        
        # Randomize initial traffic
        num_cars = random.randint(3, 8) 
        for _ in range(num_cars):
            self.spawn_vehicle(is_emergency=False, direction=random.choice(["NS", "EW"]))
            
        # Ensure exactly one ambulance spawns
        self.spawn_vehicle(is_emergency=True, direction="NS")
        
        return self.get_observation(), {}

    def spawn_vehicle(self, is_emergency=False, direction="NS"):
        # Spawn logic: randomized start position to feel "real"
        if direction == "NS":
            start_x = 5 # Fixed lane for demo
            start_y = random.randint(0, 2)
        else:
            start_x = random.randint(0, 2)
            start_y = 5 # Fixed lane for demo
            
        new_car = Vehicle(start_x, start_y, is_emergency, direction=direction)
        self.vehicles.append(new_car)

    def step(self, action):
        self.light_state = "NS_GREEN" if action == 0 else "EW_GREEN"
        reward = 0
        active_vehicles = []
        reached_goal = False

        for v in self.vehicles:
            is_green = (v['dir'] == "NS" and self.light_state == "NS_GREEN") or \
                       (v['dir'] == "EW" and self.light_state == "EW_GREEN")
            
            # Intersection Stop logic
            at_stop = (v['dir'] == "NS" and v['y'] == 4) or (v['dir'] == "EW" and v['x'] == 4)
            
            if is_green or not at_stop:
                v['speed'] = 2 if v['ev'] else 1
            else:
                v['speed'] = 0 
                
            if v['dir'] == "NS": v['y'] += v['speed']
            else: v['x'] += v['speed']

            # Rewards
            if v['ev']:
                reward += 5.0 if v['speed'] > 0 else -10.0 
                if v['x'] >= 9 or v['y'] >= 9:
                    reached_goal = True
                    reward += 100.0 # High value for reaching hospital
            elif v['speed'] == 0:
                reward -= 0.5 # Penalty for causing traffic jams

            if v['x'] < 10 and v['y'] < 10:
                active_vehicles.append(v)

        self.vehicles = active_vehicles

        # Hard Mode: Dynamic Traffic Spawning
        if self.difficulty == "hard" and random.random() < 0.2:
            self.spawn_vehicle(direction=random.choice(["NS", "EW"]), is_emergency=False)

        self.steps += 1
        done = reached_goal or self.steps >= 100
        
        return self.get_observation(), reward, done, False, {}

    def get_observation(self):
        ev_list = [v for v in self.vehicles if v.get('ev')]
        ev = ev_list[0] if ev_list else {'x': 0, 'y': 0}
        waiting = self.waiting_cars_count
        light = 0 if self.light_state == "NS_GREEN" else 1
        return np.array([ev['x'], ev['y'], waiting, light], dtype=np.float32)
    
    def get_performance_score(self, frames_taken):
        if frames_taken <= 15: return 1.0
        if frames_taken >= 100: return 0.0
        score = 1.0 - ((frames_taken - 15) / (100 - 15))
        return round(max(0.2, score), 2)