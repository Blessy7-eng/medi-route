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

    def reset(self):
        self.vehicles = [] # Clear the old traffic
        self.steps = 0     # Reset the clock
        
        # Randomly decide how many cars to start with
        num_cars = random.randint(3, 8) 
        for _ in range(num_cars):
            self.spawn_vehicle(is_emergency=False)
            
        # Always spawn 1 ambulance in a random lane
        self.spawn_vehicle(is_emergency=True)
        
        return self.get_observation()

    def spawn_vehicle(self, is_emergency=False, direction="NS"):
        # Randomize start positions
        start_y = random.randint(0, 9) 
        start_x = random.randint(0, 2) 
        
        # Pass the direction to the Vehicle creator
        new_car = Vehicle(start_x, start_y, is_emergency, direction=direction)
        self.vehicles.append(new_car)

    def step(self, action):
        # 0 = North-South Green, 1 = East-West Green
        self.light_state = "NS_GREEN" if action == 0 else "EW_GREEN"
        
        reward = 0
        # Keep track of vehicles that are still inside the 10x10 grid
        active_vehicles = []
        reached_goal = False

        # 2. Physics & Movement Loop
        for v in self.vehicles:
            # Check if the light is green for this vehicle's direction
            is_green = (v['dir'] == "NS" and self.light_state == "NS_GREEN") or \
                    (v['dir'] == "EW" and self.light_state == "EW_GREEN")
            
            # Stop line logic at index 4 (The "Intersection Entrance")
            at_stop = (v['dir'] == "NS" and v['y'] == 4) or (v['dir'] == "EW" and v['x'] == 4)
            
            # Logic: Move if it's green OR if you haven't reached the stop line yet
            if is_green or not at_stop:
                v['speed'] = 2 if v['ev'] else 1
            else:
                v['speed'] = 0 # Stuck at Red Light
                
            # Update coordinates
            if v['dir'] == "NS": 
                v['y'] += v['speed']
            else: 
                v['x'] += v['speed']

            # 3. Reward Logic (The "Brain's" Feedback)
            if v['ev']:
                # Did the ambulance move or is it stuck?
                reward += 5.0 if v['speed'] > 0 else -10.0 
                # Check if it reached the finish line
                if v['x'] >= 9 or v['y'] >= 9:
                    reached_goal = True
                    reward += 50.0 # Huge bonus for mission success!
            elif v['speed'] == 0:
                reward -= 0.2 # Small penalty for normal traffic delay

            # Only keep vehicles that are still on the grid
            if v['x'] < 10 and v['y'] < 10:
                active_vehicles.append(v)

        self.vehicles = active_vehicles

        # 4. Hard Scenario: Continuous Spawning (Sangli Market Style)
        if self.difficulty == "hard":
            if random.random() < 0.3:
                # Only spawn a normal car, never a second ambulance during a mission
                self.spawn_vehicle(direction=random.choice(["NS", "EW"]), is_emergency=False)

        # 5. Ending the Episode
        self.steps += 1
        # Terminate if goal reached or timed out (100 steps)
        done = reached_goal or self.steps >= 100
        
        obs = self.get_observation()
        
        # Return 5 values as required by Gymnasium: obs, reward, terminated, truncated, info
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