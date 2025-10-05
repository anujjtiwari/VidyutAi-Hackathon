# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pennylane as qml
import tensorflow as tf

app = Flask(__name__)

# --- 1. Load All Your Trained Models & Artifacts ---
MODEL_DIR = r"D:\Programming\VidyutAiHackathon\models"
scaler = joblib.load(f"{MODEL_DIR}/scaler1.joblib")
svc = joblib.load(f"{MODEL_DIR}/qsvm_svc1.joblib")
X_train_s = np.load(f"{MODEL_DIR}/X_train_s1.npy")
psi_train = np.load(f"{MODEL_DIR}/psi_train1.npy") # Load pre-computed training states
dqn_agent = tf.keras.models.load_model(
    f"{MODEL_DIR}/dqn_agent_model1.h5", 
    compile=False  # Add this argument
)

# --- 2. Re-create the Quantum Feature Map ---
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def psi(x):
    # Same feature map
    for i in range(n_qubits):
        qml.RY(x[i] * np.pi, wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RZ(x[i] * np.pi / 2, wires=i)
    return qml.state()

# --- 3. Define the Main Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting JSON like: {"soc": 55.2, "temp": 305.1, "volt": 3.8}
    current_state = np.array([data['soc'], data['temp'], data['volt']])

    # --- QSVM Risk Prediction ---
    # Scale the input
    scaled_state = scaler.transform(current_state.reshape(1, -1))
    
    # Compute quantum state for the new data point
    psi_x = psi(scaled_state[0])
    
    # Compute the kernel vector between the new point and all training points
    kernel_vector = np.array([np.abs(np.vdot(psi_x, pt))**2 for pt in psi_train]).reshape(1, -1)
    
    # Get risk probabilities and class from the QSVM
    risk_probs = svc.predict_proba(kernel_vector)[0]
    risk_class_index = np.argmax(risk_probs)
    risk_map = {0: 'High', 1: 'Medium', 2: 'Low'}
    risk_level = risk_map.get(risk_class_index, 'Unknown')

    # --- RL Charging Recommendation ---
    q_values = dqn_agent.predict(current_state.reshape(1, -1), verbose=0)[0]
    action_index = np.argmax(q_values)
    action_map = {0: 'Fast Charge', 1: 'Slow Charge', 2: 'Pause Charging'}
    recommendation = action_map.get(action_index, 'No Action')

    # --- Calculate Battery Health Index (BHI) ---
    # A simple formula: lower probability of high risk = higher health
    bhi = (1 - risk_probs[2]) * 100 # Assuming class 2 is 'High' risk

    # --- Construct the JSON response ---
    response = {
        'battery_health_index': round(bhi, 2),
        'risk_level': risk_level,
        'risk_probabilities': {
            'low': round(risk_probs[0], 3),
            'medium': round(risk_probs[1], 3),
            'high': round(risk_probs[2], 3)
        },
        'charging_recommendation': recommendation,
        'actionable_insight': get_insight(risk_level, recommendation, data)
    }
    
    return jsonify(response)

def get_insight(risk, rec, data):
    """Generates a human-readable reason for the recommendation."""
    if risk == 'High':
        return f"ALERT: High risk detected at {data['temp']:.1f}K! {rec} is advised to prevent damage."
    if 'Fast Charge' in rec and data['temp'] > 315:
        return f"Recommendation is {rec}, but monitor temperature closely as it's elevated."
    if 'Slow Charge' in rec:
        return "Slow charging is recommended to maintain optimal battery health and low temperature."
    return "System operating under normal conditions."

if __name__ == '__main__':
    app.run(debug=True, port=5000)