# dashboard.py
import streamlit as st
import requests
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go

# --- 1. Page & Simulation Configuration ---
st.set_page_config(
    page_title="VidyutAI EV Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Simulation parameters
CELL_GRID_SHAPE = (8, 12)

# --- Define Simulation Scenarios ---
SCENARIOS = {
    "Normal Driving": {
        "soc_change": lambda: np.random.uniform(-0.8, -0.2),
        "temp_change": lambda: np.random.uniform(-0.1, 0.2),
        "volt_change": lambda: np.random.uniform(-0.005, -0.001),
        "duration": 20,
        "message": "Status: Normal Driving Conditions"
    },
    "Aggressive Fast Charging": {
        "soc_change": lambda: np.random.uniform(1.5, 2.5),
        "temp_change": lambda: np.random.uniform(0.8, 1.5),
        "volt_change": lambda: np.random.uniform(0.01, 0.02),
        "duration": 15,
        "message": "ALERT: Aggressive Fast Charging Detected!"
    },
    "Faulty Cell Event": {
        "soc_change": lambda: np.random.uniform(-0.5, 0),
        "temp_change": lambda: np.random.uniform(0.1, 0.3),
        "volt_change": lambda: np.random.uniform(-0.01, -0.005),
        "duration": 10,
        "message": "CRITICAL ALERT: Faulty Cell - High Thermal Anomaly!"
    },
    "Regenerative Braking": {
        "soc_change": lambda: np.random.uniform(0.1, 0.3),
        "temp_change": lambda: np.random.uniform(-0.2, 0.1),
        "volt_change": lambda: np.random.uniform(0.001, 0.005),
        "duration": 8,
        "message": "Status: Regenerative Braking Active"
    }
}

# --- 2. Helper Functions ---
def get_prediction(soc, temp, volt):
    """Calls the backend API to get AI model predictions."""
    try:
        url = "http://127.0.0.1:5000/predict"
        payload = {"soc": soc, "temp": temp, "volt": volt}
        response = requests.post(url, json=payload)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None

def create_cell_grid_plot(cell_temps):
    """Creates a Plotly heatmap for the battery cell grid."""
    fig = go.Figure(data=go.Heatmap(
        z=cell_temps,
        colorscale='RdYlBu_r',
        zmin=290, zmax=340,
        showscale=False
    ))
    fig.update_layout(
        width=400, height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange='reversed'),
    )
    return fig

# --- 3. Custom CSS for Car-like UI ---
st.markdown("""
<style>
    .st-emotion-cache-1r4qj8v { background-color: #0E1117; }
    .stMetric { background-color: #262730; border-radius: 10px; padding: 15px; }
    .st-emotion-cache-176r9se p { font-size: 1.2rem; }
    .st-emotion-cache-176r9se div[data-testid="stMetricValue"] { font-size: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# --- 4. Dashboard UI Layout ---
st.title("⚡ VidyutAI - Real-time EV Monitoring")

# *** NEW: Add a sidebar for controls ***
with st.sidebar:
    st.header("Simulator Controls")
    # *** NEW: This is the button to trigger the high-risk event ***
    if st.button("Trigger High-Risk Anomaly", type="primary"):
        st.session_state.force_high_risk = True # Set a flag in session state
    st.info("Click the button to manually trigger a high-temperature fault in the battery matrix.")

status_placeholder = st.empty()
col1, col2, col3 = st.columns([1.5, 2, 1.5])

with col1:
    st.subheader("🔋 Battery Core")
    soc_placeholder = st.empty()
    volt_placeholder = st.empty()
    bhi_placeholder = st.empty()
with col2:
    st.subheader("🌡️ Thermal Matrix")
    grid_placeholder = st.empty()
with col3:
    st.subheader("💡 AI Advisory")
    risk_placeholder = st.empty()
    rec_placeholder = st.empty()

st.markdown("---")
st.subheader("📊 Historical Trends")
chart_placeholder = st.empty()

# --- 5. Main Simulation & Display Loop ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['SoC', 'Avg Temp', 'BHI'])
    st.session_state.soc = 75.0
    st.session_state.volt = 4.0
    st.session_state.cell_temps = np.full(CELL_GRID_SHAPE, 300.0)
    st.session_state.scenario_name = "Normal Driving"
    st.session_state.scenario_step = 0
    st.session_state.faulty_cell_pos = None
    st.session_state.force_high_risk = False # *** NEW: Initialize the flag ***

while True:
    # *** NEW: Check if the high-risk button was pressed ***
    if st.session_state.get("force_high_risk", False):
        st.session_state.scenario_name = "Faulty Cell Event"
        st.session_state.scenario_step = 0
        
        # Manually overheat a random cell to a critical temperature
        r, c = np.random.randint(0, CELL_GRID_SHAPE[0]), np.random.randint(0, CELL_GRID_SHAPE[1])
        st.session_state.cell_temps[r, c] = 345.0 # Set a very high temp
        st.session_state.faulty_cell_pos = (r,c)
        
        st.session_state.force_high_risk = False # Reset the flag

    scenario = SCENARIOS[st.session_state.scenario_name]
    
    # --- Update battery state based on scenario ---
    st.session_state.soc += scenario["soc_change"]()
    st.session_state.volt += scenario["volt_change"]()
    temp_delta = scenario["temp_change"]()
    st.session_state.cell_temps += temp_delta + np.random.rand(*CELL_GRID_SHAPE) * 0.1
    
    if st.session_state.scenario_name == "Faulty Cell Event":
        if st.session_state.faulty_cell_pos is None:
            r, c = np.random.randint(0, CELL_GRID_SHAPE[0]), np.random.randint(0, CELL_GRID_SHAPE[1])
            st.session_state.faulty_cell_pos = (r, c)
        r, c = st.session_state.faulty_cell_pos
        st.session_state.cell_temps[r, c] += np.random.uniform(3.0, 5.0)
    else:
        st.session_state.faulty_cell_pos = None

    # Clip values
    st.session_state.soc = np.clip(st.session_state.soc, 0, 100)
    st.session_state.volt = np.clip(st.session_state.volt, 3.2, 4.2)
    st.session_state.cell_temps = np.clip(st.session_state.cell_temps, 280, 350)

    # --- Get AI prediction from backend ---
    avg_temp = np.mean(st.session_state.cell_temps)
    api_data = get_prediction(st.session_state.soc, avg_temp, st.session_state.volt)

    # --- Update UI Elements ---
    status_placeholder.header(scenario["message"])
    soc_placeholder.metric("State of Charge", f"{st.session_state.soc:.1f}%", f"{scenario['soc_change']():+.1f}%")
    volt_placeholder.metric("Voltage", f"{st.session_state.volt:.2f}V", f"{scenario['volt_change']():+.3f}V")
    grid_placeholder.plotly_chart(create_cell_grid_plot(st.session_state.cell_temps), use_container_width=True)

    if api_data:
        bhi = api_data['battery_health_index']
        risk = api_data['risk_level']
        rec = api_data['charging_recommendation']
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk, "grey")
        bhi_delta = bhi - st.session_state.history['BHI'].iloc[-1] if not st.session_state.history.empty else 0
        bhi_placeholder.metric("Health Index (BHI)", f"{bhi:.1f}%", f"{bhi_delta:+.1f}%")
        risk_placeholder.markdown(f"**Risk Level:** <font color='{risk_color}' size='+2'>{risk}</font>", unsafe_allow_html=True)
        rec_placeholder.info(f"**Recommendation:** {rec}")
        new_row = pd.DataFrame([{'SoC': st.session_state.soc, 'Avg Temp': avg_temp, 'BHI': bhi}])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(100)
        chart_placeholder.line_chart(st.session_state.history.set_index('SoC')[['Avg Temp', 'BHI']])
    else:
        st.error("Backend API connection failed. Is app.py running?")

    # --- Switch to a new scenario after its duration ends ---
    st.session_state.scenario_step += 1
    if st.session_state.scenario_step > scenario["duration"]:
        st.session_state.scenario_name = np.random.choice(list(SCENARIOS.keys()))
        st.session_state.scenario_step = 0
        
    time.sleep(1)