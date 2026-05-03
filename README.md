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
## 🧩 Reinforcement Learning Framework

### 1. Action Space
The agent operates in a **Discrete Action Space** with 2 possible actions:
* `0`: Set North-South signals to **GREEN** (East-West becomes RED).
* `1`: Set East-West signals to **GREEN** (North-South becomes RED).

### 2. Observation Space
The agent receives a flattened state representation (Observation Vector) containing:
* **Vehicle Counts:** Number of vehicles currently waiting in each lane.
* **Ambulance Position:** The $(x, y)$ coordinates of the Emergency Vehicle.
* **Signal State:** The current phase of the traffic lights.
* **Distance to Goal:** Proximity of the ambulance to the hospital.

### 🎯 Task Descriptions
* **Easy:** Single ambulance, low traffic density (1-5 cars).
* **Medium:** Mixed traffic flow, ambulance starting at a random intersection.
* **Hard (Sangli Rush):** Max density (20+ cars), randomized spawning, and complex arrival patterns.

---

## 📊 Results
Our agent was tested across three rigorous scenarios:
1. **Easy:** Clear roads to establish a baseline.
2. **Medium:** Standard daytime traffic.
3. **Hard (Sangli Market Rush):** High-density, randomized vehicle spawning.

**The Result:** The AI achieved a **1.0 Perfect Score**, reducing ambulance wait time to nearly zero while maintaining a steady flow for 80% of civilian traffic.

---

## 📥 Installation & Local Setup
If you want to run Medi-Route on your local machine, follow these steps:

1. **Clone the Repository**

Bash
```
git clone https://github.com/Blessy7-eng/medi-route.git
cd medi-route
```

2. **Manual Installation (Python 3.10+)**
It is recommended to use a virtual environment to avoid dependency conflicts.

Bash
```
# Create a virtual environment
python -m venv venv
```
# Activate it (Windows)
```
venv\Scripts\activate
```
# Activate it (Linux/Mac)
```
source venv/bin/activate
```
# Install dependencies
```
pip install -r requirements-deploy.txt
```
3. **Running the API Server**
To test the validator-style API locally:

Bash
```
python inference.py
```
The API will be available at http://localhost:7860. You can test the health check at http://localhost:7860/health.

4. **Running the Dashboard (Streamlit)**
To visualize the traffic simulation:

Bash
```
streamlit run ui.py
```
🐳 Running with Docker
This is the preferred method for deployment and matches the Meta Hackathon environment.

**Build the image:**

Bash
```
docker build -t medi-route .
```
**Run the container:**

Bash
```
docker run -p 7860:7860 medi-route
```
**🧪 API Testing**
Once the server is running, you can interact with the RL environment using curl or Postman:

**Reset the Environment:**


```
curl -X POST http://localhost:7860/reset
```
Take an Action (0 for North-South Green, 1 for East-West Green):

```
curl -X POST "http://localhost:7860/step?action=0"
```
3. Srushti Sanjay Shirale
| **Location:** Sangli, Maharashtra
