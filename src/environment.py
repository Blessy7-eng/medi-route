import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def Vehicle(x, y, is_emergency=False, direction="NS"):
    """
    Creates a vehicle dictionary compatible with your current step() logic.
    """
    return {
        'x': x,
        'y': y,
        'ev': is_emergency,
        'dir': direction,
        'speed': 0
    }

class TrafficEnv(gym.Env):
    @property
    def ambulance_speed_kmh(self):
        amb = [v for v in self.vehicles if v.get('ev')]
        if not amb: return 0
        return 60 if amb[0]['speed'] > 0 else 0

    @property
    def waiting_cars_count(self):
        return len([v for v in self.vehicles if v['speed'] == 0 and not v.get('ev')])
    
    def __init__(self, difficulty="medium"):
        super(TrafficEnv, self).__init__()
        self.grid_size = 10
        self.difficulty = difficulty
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 20, 1], dtype=np.float32),
            dtype=np.float32
        )
        # We don't call reset() here anymore, __init__ should just setup.
        # But for compatibility with your app.py, let's ensure reset is safe.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicles = [] 
        self.steps = 0     
        
        # --- FIX: Initialize light_state BEFORE calling get_observation ---
        self.light_state = "NS_GREEN" 
        
        num_cars = random.randint(3, 8) 
        for _ in range(num_cars):
            self.spawn_vehicle(is_emergency=False, direction=random.choice(["NS", "EW"]))
            
        self.spawn_vehicle(is_emergency=True, direction="NS")
        
        return self.get_observation(), {}

    def spawn_vehicle(self, is_emergency=False, direction="NS"):
        start_y = random.randint(0, 9) 
        start_x = random.randint(0, 2) 
        new_car = Vehicle(start_x, start_y, is_emergency, direction=direction)
        self.vehicles.append(new_car)

    def step(self, action):
        # 0 = North-South Green, 1 = East-West Green
        self.light_state = "NS_GREEN" if action == 0 else "EW_GREEN"
        
        reward = 0
        active_vehicles = []
        reached_goal = False

        for v in self.vehicles:
            is_green = (v['dir'] == "NS" and self.light_state == "NS_GREEN") or \
                    (v['dir'] == "EW" and self.light_state == "EW_GREEN")
            
            at_stop = (v['dir'] == "NS" and v['y'] == 4) or (v['dir'] == "EW" and v['x'] == 4)
            
            if is_green or not at_stop:
                v['speed'] = 2 if v['ev'] else 1
            else:
                v['speed'] = 0 
                
            if v['dir'] == "NS": 
                v['y'] += v['speed']
            else: 
                v['x'] += v['speed']

            if v['ev']:
                reward += 5.0 if v['speed'] > 0 else -10.0 
                if v['x'] >= 9 or v['y'] >= 9:
                    reached_goal = True
                    reward += 50.0 
            elif v['speed'] == 0:
                reward -= 0.2 

            if v['x'] < 10 and v['y'] < 10:
                active_vehicles.append(v)

        self.vehicles = active_vehicles

        if self.difficulty == "hard":
            if random.random() < 0.3:
                self.spawn_vehicle(direction=random.choice(["NS", "EW"]), is_emergency=False)

        self.steps += 1
        done = reached_goal or self.steps >= 100
        
        obs = self.get_observation()
        return obs, reward, done, False, {}

    def get_observation(self):
        ev_list = [v for v in self.vehicles if v['ev']]
        ev = ev_list[0] if ev_list else {'x': 0, 'y': 0}
        waiting = self.waiting_cars_count
        # This will no longer crash because light_state is set in reset()
        light = 0 if self.light_state == "NS_GREEN" else 1
        return np.array([ev['x'], ev['y'], waiting, light], dtype=np.float32)
    
    def get_performance_score(self, frames_taken):
        if frames_taken <= 15:
            return 1.0
        elif frames_taken >= 100:
            return 0.0
        else:
            score = 1.0 - ((frames_taken - 15) / (100 - 15))
            return round(max(0.2, score), 2)