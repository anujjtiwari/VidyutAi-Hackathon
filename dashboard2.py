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

# --- 2. Custom CSS for a sleek, car-like UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Set a dark background for the main content area */
    section[data-testid="st.main"] {
        background-color: #0E1117;
    }
    
    /* Main title */
    h1 {
        color: #FFFFFF;
        font-size: 2.5rem;
    }

    /* Section headers */
    h2 {
        color: #A0AEC0;
        font-weight: 600;
        border-bottom: 2px solid #2D3748;
        padding-bottom: 8px;
    }

    /* Metric cards styling */
    .metric-card {
        background-color: #1A202C;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2D3748;
        margin-bottom: 1rem;
    }
    .metric-card .label {
        color: #A0AEC0;
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-card .value {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-card .unit {
        color: #718096;
        font-size: 1.5rem;
        margin-left: 8px;
    }

    /* AI Advisory card */
    .advisory-card {
        background-color: #1A202C;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid; /* Color will be set dynamically */
    }
    .advisory-card .title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .advisory-card .recommendation {
        font-size: 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Simulation Scenarios ---
SCENARIOS = {
    "Normal Driving": {
        "soc_change": lambda: np.random.uniform(-0.8, -0.2),
        "temp_change": lambda: np.random.uniform(-0.1, 0.2),
        "message": "SYSTEMS NOMINAL",
        "color": "#38A169" # Green
    },
    "Aggressive Fast Charging": {
        "soc_change": lambda: np.random.uniform(1.5, 2.5),
        "temp_change": lambda: np.random.uniform(0.8, 1.5),
        "message": "WARNING: HIGH THERMAL LOAD",
        "color": "#DD6B20" # Orange
    },
    "Faulty Cell Event": {
        "soc_change": lambda: np.random.uniform(-0.5, 0),
        "temp_change": lambda: np.random.uniform(0.1, 0.3),
        "message": "CRITICAL ALERT: THERMAL ANOMALY DETECTED",
        "color": "#E53E3E" # Red
    }
}
CELL_GRID_SHAPE = (8, 12)

# --- 4. Helper Functions ---
def get_prediction(soc, temp, volt):
    """Calls the backend API to get AI model predictions."""
    try:
        url = "http://127.0.0.1:5000/predict"
        payload = {"soc": soc, "temp": temp, "volt": volt}
        response = requests.post(url, json=payload, timeout=0.5) # 0.5 second timeout
        return response.json() if response.status_code == 200 else None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return None

def create_bhi_gauge(bhi_value):
    """Creates a sleek Plotly gauge for the Battery Health Index."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bhi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "#1A202C",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#E53E3E'},
                {'range': [40, 70], 'color': '#DD6B20'},
                {'range': [70, 100], 'color': '#38A169'}],
            'threshold': {
                'line': {'color': "white", 'width': 5},
                'thickness': 0.85,
                'value': bhi_value
            }
        },
        number={'suffix': "%", 'font': {'size': 50, 'color': 'white'}}
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0E1117",
        font={'color': 'white'}
    )
    return fig

def create_cell_grid_plot(cell_temps):
    """Creates a beautiful, green-themed Plotly heatmap for the battery cell grid."""
    # Custom green-to-red colorscale
    colorscale = [
        [0.0, '#0d4b24'],    # Dark Green (Cool)
        [0.5, '#24ff88'],    # Neon Green (Normal)
        [0.75, '#F9F871'],   # Yellow (Warning)
        [1.0, '#E53E3E']     # Red (Critical)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cell_temps,
        colorscale=colorscale,
        zmin=290, zmax=340,
        showscale=False
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1A202C',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange='reversed'),
    )
    return fig

# --- 5. Dashboard UI Layout ---

st.title("VIDYUTAI OS")
st.markdown("---")

# Main Status Bar
status_placeholder = st.empty()

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("<h2>Battery Health Index (BHI)</h2>", unsafe_allow_html=True)
    bhi_gauge_placeholder = st.empty()
    
    st.markdown("<h2>Core Vitals</h2>", unsafe_allow_html=True)
    soc_placeholder = st.empty()
    volt_placeholder = st.empty()

with col2:
    st.markdown("<h2>AI Advisory System</h2>", unsafe_allow_html=True)
    advisory_placeholder = st.empty()
    
    st.markdown("<h2>Thermal Matrix</h2>", unsafe_allow_html=True)
    grid_placeholder = st.empty()

# Sidebar for controls
with st.sidebar:
    st.header("Simulator Controls")
    scenario_selection = st.selectbox(
        "Choose a simulation scenario:",
        list(SCENARIOS.keys())
    )

# --- 6. Main Simulation & Display Loop ---

# Initialize state if it doesn't exist
if 'soc' not in st.session_state:
    st.session_state.soc = 75.0
    st.session_state.volt = 4.0
    st.session_state.cell_temps = np.full(CELL_GRID_SHAPE, 300.0)

# This loop provides a smooth update but may be unstable and could cause a
# 'StreamlitDuplicateElementId' error after running for a while.
while True:
    scenario = SCENARIOS[scenario_selection]

    # Update state based on the selected scenario
    st.session_state.soc += scenario["soc_change"]()
    st.session_state.cell_temps += scenario["temp_change"]() + np.random.rand(*CELL_GRID_SHAPE) * 0.1
    st.session_state.volt = 4.0 + (st.session_state.soc - 75) * 0.008 + np.random.uniform(-0.01, 0.01)

    # Specific logic for Faulty Cell scenario
    if scenario_selection == "Faulty Cell Event":
        r, c = (3, 6) # Make a specific cell faulty for consistency
        st.session_state.cell_temps[r, c] += np.random.uniform(3.0, 5.0)

    # Clip values to stay within realistic bounds
    st.session_state.soc = np.clip(st.session_state.soc, 0, 100)
    st.session_state.volt = np.clip(st.session_state.volt, 3.2, 4.2)
    st.session_state.cell_temps = np.clip(st.session_state.cell_temps, 280, 350)

    # Get AI prediction from the backend
    avg_temp = np.mean(st.session_state.cell_temps)
    api_data = get_prediction(st.session_state.soc, avg_temp, st.session_state.volt)

    # --- Update UI Elements ---

    # Status bar
    status_placeholder.markdown(
        f'<div style="background-color: {scenario["color"]}; color: white; text-align: center; padding: 10px; border-radius: 8px; font-size: 1.5rem; font-weight: 700;">'
        f'{scenario["message"]}</div>',
        unsafe_allow_html=True
    )

    # Core Vitals
    soc_placeholder.markdown(
        f'<div class="metric-card"><div class="label">State of Charge</div><span class="value">{st.session_state.soc:.1f}</span><span class="unit">%</span></div>',
        unsafe_allow_html=True
    )
    volt_placeholder.markdown(
        f'<div class="metric-card"><div class="label">Voltage</div><span class="value">{st.session_state.volt:.2f}</span><span class="unit">V</span></div>',
        unsafe_allow_html=True
    )

    # Thermal Matrix
    grid_placeholder.plotly_chart(create_cell_grid_plot(st.session_state.cell_temps), use_container_width=True)

    # AI Advisory & BHI Gauge (driven by backend data)
    if api_data:
        bhi = api_data['battery_health_index']
        risk = api_data['risk_level']
        
        # --- NEW LOGIC: Determine recommendation based on risk level ---
        if risk == "High":
            display_rec = "Pause Charging Immediately"
        elif risk == "Medium":
            display_rec = "Slow Charging Recommended"
        else: # Low risk
            display_rec = "Fast Charging Approved"
        
        bhi_gauge_placeholder.plotly_chart(create_bhi_gauge(bhi), use_container_width=True)
        
        risk_color = {"Low": "#38A169", "Medium": "#DD6B20", "High": "#E53E3E"}.get(risk, "#A0AEC0")
        
        advisory_placeholder.markdown(
            f'<div class="advisory-card" style="border-left-color: {risk_color};">'
            f'<div class="title" style="color: {risk_color};">RISK LEVEL: {risk.upper()}</div>'
            f'<div class="recommendation" style="color: white;">{display_rec}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # Show a placeholder if the backend isn't connected
        bhi_gauge_placeholder.plotly_chart(create_bhi_gauge(0), use_container_width=True)
        advisory_placeholder.markdown(
            f'<div class="advisory-card" style="border-left-color: #A0AEC0;">'
            f'<div class="title" style="color: #A0AEC0;">RISK LEVEL: UNKNOWN</div>'
            f'<div class="recommendation" style="color: white;">Connecting to AI Core...</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    time.sleep(1)

