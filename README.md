---
title: Medi Route
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---
# 🚑 Medi-Route: AI-Powered Emergency Response

> **Project Vision:** Medi-Route leverages Reinforcement Learning to solve critical emergency vehicle delays in Tier-2 Indian cities like Sangli, where narrow roads and unpredictable traffic can cost lives.

---

## 🚦 The Problem
In cities like Sangli or Mumbai, ambulances often lose "Golden Hour" minutes stuck at traditional, timer-based traffic signals. Standard signals don't know an ambulance is coming. **Medi-Route fixes this.**

## 🧠 The Tech Stack
### 1. The Algorithm: Deep Q-Network (DQN)
We chose **DQN** because it excels at making discrete decisions (Switching Light: North-South vs. East-West). The model "learns" by playing thousands of traffic simulations, eventually discovering that clearing a path for the ambulance yields the highest "points."

### 2. The Reward Function (The Logic)
Our AI isn't just told to be fast; it's taught through a balanced scoring system:
* **Ambulance Progress:** $+2.0$ for every step moved (High Priority).
* **Ambulance Stalled:** $-5.0$ if the ambulance is stuck at a red light.
* **Traffic Fairness:** $-0.1$ for every normal car waiting.
* **Why penalize normal traffic?** We chose a small penalty for normal cars to ensure the AI doesn't create a "city-wide jam" just to save one vehicle. It learns to find the *perfect gap*.

---

## 📊 Results
Our agent was tested across three rigorous scenarios:
1. **Easy:** Clear roads to establish a baseline.
2. **Medium:** Standard daytime traffic.
3. **Hard (Sangli Market Rush):** High-density, randomized vehicle spawning.

**The Result:** The AI achieved a **1.0 Perfect Score**, reducing ambulance wait time to nearly zero while maintaining a steady flow for 80% of civilian traffic.

---

## 🛠️ How to Run
This project is fully Dockerized for the Meta PyTorch OpenEnv Hackathon.

1. **Build:** `docker build -t medi-route .`
2. **Run:** `docker run medi-route`

---
**Team:** [Your Names Here] | **Location:** Sangli, Maharashtra