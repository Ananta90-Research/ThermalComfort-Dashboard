import streamlit as st
import pandas as pd
import numpy as np
import joblib
import copy

# ---------------- Load Model ----------------
model = joblib.load("ThermalComfort_prediction_model.pkl")

# ---------------- Base Databases ----------------
city_weather = {
    "Mumbai": {"Temperature": 31, "SolarFlux": 932, "Humidity": 65, "WindSpeed": 4, "CloudCoverage": 5},
    "Jodhpur": {"Temperature": 40, "SolarFlux": 925, "Humidity": 25, "WindSpeed": 1, "CloudCoverage": 5},
    "Bengaluru": {"Temperature": 28, "SolarFlux": 946, "Humidity": 54, "WindSpeed": 3, "CloudCoverage": 3},
}

glass_props = {
    "Windshield": {
        "TSANx//TSANx": {"Te": 0.536, "Tts": 0.647},
        "TSA3+//TSA3+": {"Te": 0.425, "Tts": 0.568},
        "TSANx//TSA3+": {"Te": 0.449, "Tts": 0.585},
        "CLR(IRR coating)//TSANx": {"Te": 0.293, "Tts": 0.367},
    },
    "Sidelite": {
        "TSANx": {"Te": 0.634, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "Tts": 0.619},
        "TSA5+": {"Te": 0.360, "Tts": 0.522},
        "Inshade": {"Te": 0.401, "Tts": 0.521},
    },
    "Backlite": {
        "TSANx": {"Te": 0.634, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "Tts": 0.619},
        "TSA5+": {"Te": 0.360, "Tts": 0.522},
    },
    "Roof": {
        "VG10": {"Te": 0.1, "Tts": 0.336},
        "VG20": {"Te": 0.165, "Tts": 0.383},
        "CLR(IRR coating)//VG10": {"Te": 0.101, "Tts": 0.230},
    },
}

# ---------------- Session State Initialization ----------------
if "glass_props_session" not in st.session_state:
    st.session_state.glass_props_session = copy.deepcopy(glass_props)

if "city_weather_session" not in st.session_state:
    st.session_state.city_weather_session = copy.deepcopy(city_weather)

if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# ---------------- Page Config ----------------
st.set_page_config(page_title="🚗 Thermal Comfort Predictor", layout="wide")
st.title("🚗 Thermal Comfort Dashboard")

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
    .big-label {
        font-size:20px !important;
        font-weight:bold !important;
    }
    .prediction-text {
        font-size:18px !important;
    }
    .history-table td, .history-table th {
        font-size:14px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Layout ----------------
col1, spacer, col2 = st.columns([2, 0.1, 2.5])

# ---------------- Column 1: Inputs ----------------
with col1:
    st.markdown("<p class='big-label'>🏙️ Select City</p>", unsafe_allow_html=True)
    city_options = list(st.session_state.city_weather_session.keys()) + ["➕ Add Custom Weather"] 
    city = st.selectbox("", city_options, key="city_select")

    if city == "➕ Add Custom Weather":
        new_city_name = st.text_input("Enter new city name")
        temp = st.slider("🌡️ Ambient Temperature (°C)", 20, 50, 35)
        solar = st.slider("☀️ Solar Flux (W/m²)", 500, 1200, 900)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 50)
        wind = st.slider("🌬️ Wind Speed (m/s)", 0, 10, 5)
        cloud = st.slider("☁️ Cloud Coverage (0-10)", 0, 10, 5)

        if st.button("✅ Add City"):
            if not new_city_name:
                st.warning("⚠️ Please enter a city name.")
            elif new_city_name in st.session_state.city_weather_session:
                st.warning("⚠️ City already exists.")
            else:
                st.session_state.city_weather_session[new_city_name] = {
                    "Temperature": temp,
                    "SolarFlux": solar,
                    "Humidity": humidity,
                    "WindSpeed": wind,
                    "CloudCoverage": cloud,
                }
                st.success(f"✅ City '{new_city_name}' added.")
                st.experimental_rerun()
    else:
        weather = st.session_state.city_weather_session[city]

    # ---------------- Glass Selector ----------------
    def glass_selector(position):
        st.markdown(f"<p class='big-label'>{position} Glass Type</p>", unsafe_allow_html=True)
        glass_list = list(st.session_state.glass_props_session[position].keys()) + ["➕ Add New Glass Type"]
        selected = st.selectbox(f"{position} Glass Type", glass_list, key=position)

        if selected == "➕ Add New Glass Type":
            with st.expander(f"➕ Add New Glass Type to {position}"):
                new_name = st.text_input("Glass Name", key=f"{position}_name")
                new_trans = st.slider("Transmittance (Te)", 0.01, 0.9, 0.6, key=f"{position}_Te")
                new_Tts = st.slider("Transmitted Solar Energy (Tts)", 0.0, 0.7, 0.3, key=f"{position}_Tts")

                if st.button("➕ Add", key=f"{position}_add"):
                    if new_name and new_name not in st.session_state.glass_props_session[position]:
                        st.session_state.glass_props_session[position][new_name] = {
                            "Te": new_trans,
                            "Tts": new_Tts,
                        }
                        st.success(f"✅ Added '{new_name}' to {position}")
                        st.experimental_rerun()
                    else:
                        st.warning("⚠️ Name exists or is empty. Please enter a unique name.")

        props = st.session_state.glass_props_session[position].get(selected, None)
        if props is None:
            fallback = list(st.session_state.glass_props_session[position].values())[0]
            return fallback["Te"], fallback["Tts"]
        return props["Te"], props["Tts"]

    # Get properties for all glasses
    Te_ws, Tts_ws = glass_selector("Windshield")
    Te_sl, Tts_sl = glass_selector("Sidelite")
    Te_bl, Tts_bl = glass_selector("Backlite")
    Te_roof, Tts_roof = glass_selector("Roof")

# ---------------- Column 2: Prediction ----------------
with col2:
    st.header("📊 Prediction Results")

    input_row = pd.DataFrame([{
        "SolarFlux": weather["SolarFlux"],
        "Temperature": weather["Temperature"],
        "WindSpeed": weather["WindSpeed"],
        "CloudCoverage": weather["CloudCoverage"],
        "Humidity": weather["Humidity"],
        "Te(WS)": Te_ws, "Tts(WS)": Tts_ws,
        "Te(SL)": Te_sl, "Tts(SL)": Tts_sl,
        "Te(BL)": Te_bl, "Tts(BL)": Tts_bl,
        "Te(roof)": Te_roof, "Tts(roof)": Tts_roof
    }])

    if st.button("🔮 Predict Cabin Temperature"):
        prediction = model.predict(input_row)[0]
        pred_value = round(prediction, 2)
        st.session_state.pred_history.append(pred_value)
        st.markdown(f"<p class='prediction-text'>🌡️ Predicted Cabin Temperature: <b>{pred_value} °C</b></p>", unsafe_allow_html=True)

    st.subheader("📦 Prediction History")

    col_h1, col_h2 = st.columns([1, 3])
    with col_h1:
        if st.button("🗑️ Clear History"):
            st.session_state.pred_history = []

    if st.session_state.pred_history:
        hist_df = pd.DataFrame(st.session_state.pred_history, columns=["Cabin_Air_Temperature (°C)"])
        st.markdown(hist_df.to_html(index=False, classes="history-table"), unsafe_allow_html=True)
    else:
        st.write("No predictions yet.")

