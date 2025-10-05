# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pennylane as qml
import tensorflow as tf

app = Flask(__name__)

# --- 1. Load Trained Models & Artifacts ---
MODEL_DIR = r"D:\Programming\VidyutAiHackathon\models"
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")          # original 3-feature scaler
svc = joblib.load(f"{MODEL_DIR}/qsvm_svc.joblib")
psi_train = np.load(f"{MODEL_DIR}/psi_train.npy")           # precomputed quantum states
dqn_agent = tf.keras.models.load_model(f"{MODEL_DIR}/dqn_agent_model.h5", compile=False)

# --- 2. Nominal Curve & BHI (reporting only) ---
nominal_curve = {100:4.01,90:3.96,80:3.92,70:3.88,60:3.83,50:3.78,40:3.73,30:3.68,20:3.62,10:3.55,0:3.45}

def compute_bhi(soc, voltage):
    soc_bucket = int((soc // 10) * 10)
    if soc_bucket not in nominal_curve:
        soc_bucket = min(nominal_curve.keys(), key=lambda k: abs(k - soc_bucket))
    expected_v = nominal_curve[soc_bucket]
    diff = abs(expected_v - voltage)
    bhi = max(0, 1 - diff / expected_v)
    return round(bhi, 4)

# --- 3. Quantum Feature Map (3 qubits) ---
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def psi(x):
    for i in range(n_qubits):
        qml.RY(x[i] * np.pi, wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RZ(x[i] * np.pi / 2, wires=i)
    return qml.state()

# --- 4. Insight helper ---
def get_insight(risk, rec, data):
    if risk == 'HIGH':
        return f"ALERT: High risk detected at {data['temp']:.1f}K! {rec} is advised to prevent damage."
    if 'Fast Charge' in rec and data['temp'] > 315:
        return f"Recommendation is {rec}, but monitor temperature closely as it's elevated."
    if 'Slow Charge' in rec:
        return "Slow charging is recommended to maintain optimal battery health and low temperature."
    return "System operating under normal conditions."

# --- 5. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # JSON format: {"soc": 55.2, "temp": 305.1, "volt": 3.8}
    
    # --- Compute BHI for reporting only ---
    bhi = compute_bhi(data['soc'], data['volt'])
    
    # --- QSVM Risk Prediction (3 features) ---
    current_state = np.array([data['soc'], data['temp'], data['volt']])
    scaled_state = scaler.transform(current_state.reshape(1, -1))
    
    # Compute quantum state
    psi_x = psi(scaled_state[0])
    
    # Correct kernel vector computation using precomputed psi_train
    kernel_vector = np.array([np.abs(np.vdot(psi_x, pt))**2 for pt in psi_train]).reshape(1, -1)
    
    # Get probabilities and predicted class
    risk_probs = svc.predict_proba(kernel_vector)[0]
    # Map class indices to labels correctly
    class_order = svc.classes_  # should be [0,1,2]
    risk_map = {class_order[i]: label for i,label in enumerate(['LOW','MEDIUM','HIGH'])}
    risk_class_index = np.argmax(risk_probs)
    risk_level = risk_map.get(class_order[risk_class_index], 'Unknown')
    
    # --- RL Charging Recommendation ---
    q_values = dqn_agent.predict(current_state.reshape(1, -1), verbose=0)[0]
    action_index = np.argmax(q_values)
    action_map = {0:'Fast Charge', 1:'Slow Charge', 2:'Pause Charging'}
    recommendation = action_map.get(action_index, 'No Action')
    
    # --- Construct JSON response ---
    response = {
        'battery_health_index': round(bhi*100,2),   # reporting only
        'risk_level': risk_level,
        'risk_probabilities': {
            'low': round(risk_probs[0],3),
            'medium': round(risk_probs[1],3),
            'high': round(risk_probs[2],3)
        },
        'charging_recommendation': recommendation,
        'actionable_insight': get_insight(risk_level, recommendation, data)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
