Team name : Watt 

QUICK SETUP:
- THE model-save folder contains joblib files for the model trained you just need to clone the repo and change the path for the directory in app.py file 
- after changing the directory path in local machine simply run the app.py if it says "Running on http://127.0.0.1:5000" it is succesfully running
- at the end just run the code "streamlit run dashboard2.py" and you can run the dashboard smoothly! :)
---

VidyutAi-Hackathon ⚡

EV-AI — Intelligent Energy & Battery Safety System

This repository contains the solution for the VidyutAi Hackathon project, focusing on predicting EV battery safety risks, computing a Battery Health Index (BHI), and recommending optimal charging/discharging strategies using AI / Reinforcement Learning.

🧩 Collaborators

Vatsal

Vishakh

Anuj

📂 Repository Structure
VidyutAi-Hackathon/
│
├── data/                         # Raw & processed datasets (Full-Data.csv, etc.)
├── model-save/                   # Trained models & artifact files (scaler, QSVM, DQN, psi_train)
├── notebooks/                     # Exploratory Data Analysis, model training, experiments
│    ├── DataCreator.ipynb
│    ├── Main.ipynb
│    └── …  
├── app.py                         # Main API / web interface
├── dashboard2.py                  # Dashboard / UI frontend
├── extra-codes/                   # Auxiliary scripts
└── README.md                      # This file

📘 Overview & Motivation

India is rapidly adopting EVs, but safety, reliability, and battery degradation remain key challenges. Factors such as thermal runaway from fast charging, environment stress (heat, humidity), and inconsistent diagnostics in charging infrastructure make it hard to build user confidence.

This project addresses these challenges by building a data-driven backend that:

Predicts battery safety risk levels (Low / Medium / High)

Computes a Battery Health Index (BHI) in real time

Detects anomalies (voltage drops, rapid temperature rise)

Uses an RL agent to recommend charging actions (fast, slow, pause)

Integrates everything into a dashboard / advisory UI

🛠️ How It Works
1. Data Preparation & Sampling

We balance the dataset across risk classes (healthy, over-discharge, short-circuit), and pick features: SoC, Temperature, Voltage for model input.

2. Quantum SVM (QSVM)

We embed feature vectors into quantum states using parameterized rotations and entangling gates, compute a quantum kernel similarity matrix, and train an SVM on that kernel. This model predicts the risk class of new battery states.

3. Battery Health Index (BHI)

For each sample, we compute a “nominal” voltage based on SoC (via interpolation) and compare it to the actual voltage:

<img width="428" height="129" alt="image" src="https://github.com/user-attachments/assets/08170fb6-9674-4e61-899a-0d1b7b79863a" />


This gives a percentage measure of battery health relative to expected behavior.

4. Reinforcement Learning Agent

We simulate a battery environment (state evolution of SoC, temperature, voltage) for actions: fast charge, slow charge, pause.
The QSVM model is used inside the environment to estimate risk, and a reward function encourages safe behavior (e.g., penalizes overheating, deep discharge).
A DQN agent learns which action to take in each state to maximize long-term safety and battery health.

5. Dashboard & API

Once models are trained, we save artifacts (scaler, QSVM, psi_train, DQN model). A Flask / Dash / Streamlit interface loads these, takes input readings (SoC, Temp, Voltage), and returns:

The computed BHI

Risk Level (Low / Medium / High)

Risk probabilities

Charging recommendation (fast / slow / pause)

Actionable insight / advisory message

The dashboard visualizes these in charts and user-friendly UI.

🏁 Quickstart

Clone the repository:

git clone https://github.com/anujjtiwari/VidyutAi-Hackathon.git
cd VidyutAi-Hackathon


(Optional) Create a virtual environment:

python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS / Linux


Install dependencies:

pip install -r requirements.txt


Train or load pre-trained models:

Use notebooks in notebooks/ to preprocess, train, evaluate.

Models will be saved in model-save/.

Run the web / dashboard app:

python app.py


or for dashboard:

streamlit run dashboard2.py


Open your browser at the displayed URL (e.g. http://127.0.0.1:5000) and interact with the UI.

🔮 Future Enhancements (Ideas)

Expand feature set: include current, humidity, load, etc.

Improve QSVM: explore deeper quantum circuits, better embeddings.

RL: incorporate longer horizons, more actions, charging station constraints.

Real-time data ingestion from EV sensors.

Deploy dashboard to mobile / tablet / embedded device.

Add comparisons, degradation curves, historical tracking.

Add unit tests, CI/CD integration, versioning.

📞 Contact & Contribution

Maintained by: Vatsal, Vishakh, Anuj

Contributions are welcome!

Fork the repo

Create your feature branch: git checkout -b feat/whatever

Commit your changes

Push and open a Pull Request

Let’s accelerate electric mobility together 🔋🚗
