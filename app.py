import streamlit as st
import pandas as pd
import numpy as np
import joblib
import copy

# ---------------- Load Model ----------------
try:
    model = joblib.load("ThermalComfort_prediction_model.pkl")
except FileNotFoundError:
    st.error("Model file 'ThermalComfort_prediction_model.pkl' not found!")
    st.stop()

# ---------------- Base Databases ----------------
city_weather = {
    "Mumbai": {"Temperature": 31, "SolarFlux": 932, "Humidity": 65, "WindSpeed": 4, "CloudCoverage": 5},
    "Jodhpur": {"Temperature": 40, "SolarFlux": 925, "Humidity": 25, "WindSpeed": 1, "CloudCoverage": 5},
    "Bengaluru": {"Temperature": 28, "SolarFlux": 946, "Humidity": 54, "WindSpeed": 3, "CloudCoverage": 3},
}

glass_props = {
    "Windshield": {
        "TSANx//TSANx": {"Te": 0.536, "t": 5.0, "Tts": 0.647},
        "TSA3+//TSA3+": {"Te": 0.425, "t": 5.0, "Tts": 0.568},
        "TSANx//TSA3+": {"Te": 0.449, "t": 5.0, "Tts": 0.585},
        "CLR(IRR coating)//TSANx": {"Te": 0.293, "t": 5.0, "Tts": 0.367},
    },
    "Sidelite": {
        "TSANx": {"Te": 0.634, "t": 3.2, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "t": 3.2, "Tts": 0.619},
        "TSA5+": {"Te": 0.360, "t": 3.2, "Tts": 0.522},
        "Inshade": {"Te": 0.401, "t": 3.2, "Tts": 0.521},
    },
    "Backlite": {
        "TSANx": {"Te": 0.634, "t": 3.2, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "t": 3.2, "Tts": 0.619},
        "TSA5+": {"Te": 0.360, "t": 3.2, "Tts": 0.522},
    },
    "Roof": {
        "VG10": {"Te": 0.1, "t": 3.85, "Tts": 0.336},
        "VG20": {"Te": 0.165, "t": 3.85, "Tts": 0.383},
        "CLR(IRR coating)//VG10": {"Te": 0.101, "t": 5.0, "Tts": 0.230},
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
st.set_page_config(page_title="Thermal comfort Predictor", layout="centered")
st.title("🚗 Thermal Comfort Dashboard")

# ---------------- City Selector ----------------
city_options = list(st.session_state.city_weather_session.keys()) + ["➕ Add Custom Weather"]
city = st.selectbox("Select City or Add Custom", city_options)

if city == "➕ Add Custom Weather":
    new_city_name = st.text_input("Enter new city name")
    temp = st.slider("Ambient Temperature (°C)", 20, 50, 35)
    solar = st.slider("Solar Flux (W/m²)", 500, 1200, 900)
    humidity = st.slider("Humidity (%)", 10, 100, 50)
    wind = st.slider("Wind Speed (km/h)", 0, 30, 10)
    cloud = st.slider("Cloud Coverage (%)", 0, 100, 20)

    if st.button("Add City"):
        if not new_city_name:
            st.warning("Please enter a city name.")
        elif new_city_name in st.session_state.city_weather_session:
            st.warning("City already exists.")
        else:
            st.session_state.city_weather_session[new_city_name] = {
                "Temperature": temp,  # Fixed typo
                "SolarFlux": solar,
                "Humidity": humidity,
                "WindSpeed": wind,
                "CloudCoverage": cloud,
            }
            st.success(f"City '{new_city_name}' added.")
            st.rerun()  # Fixed deprecated method
    
    # Set default weather for prediction if no city is selected yet
    weather = {
        "Temperature": temp,
        "SolarFlux": solar,
        "Humidity": humidity,
        "WindSpeed": wind,
        "CloudCoverage": cloud,
    }
else:
    weather = st.session_state.city_weather_session[city]

# ---------------- Glass Selector ----------------
def glass_selector(position):
    glass_list = list(st.session_state.glass_props_session[position].keys()) + ["➕ Add New Glass Type"]
    selected = st.selectbox(f"{position} Glass Type", glass_list, key=position)

    if selected == "➕ Add New Glass Type":
        with st.expander(f"Add New Glass Type to {position}"):
            new_name = st.text_input("Glass Name", key=f"{position}_name")
            new_trans = st.slider("Transmittance (Te)", 0.0, 1.0, 0.6, key=f"{position}_Te")
            new_Tts = st.slider("Transmitted Solar Energy (Tts)", 0.0, 1.0, 0.3, key=f"{position}_Tts")
            
            # Add thickness input for completeness
            default_thickness = 3.2 if position in ["Sidelite", "Backlite"] else 5.0 if position == "Windshield" else 3.85
            new_thickness = st.slider("Thickness (t)", 1.0, 10.0, default_thickness, key=f"{position}_t")

            if st.button("Add", key=f"{position}_add"):
                if new_name and new_name not in st.session_state.glass_props_session[position]:
                    st.session_state.glass_props_session[position][new_name] = {
                        "Te": new_trans,
                        "t": new_thickness,
                        "Tts": new_Tts,
                    }
                    st.success(f"✅ Added '{new_name}' to {position}")
                    st.rerun()  # Fixed deprecated method
                else:
                    st.warning("Name exists or is empty. Please enter a unique name.")

    # Return selected or fallback with error handling
    try:
        if selected != "➕ Add New Glass Type" and selected in st.session_state.glass_props_session[position]:
            props = st.session_state.glass_props_session[position][selected]
            return props.get("Te", 0.5), props.get("Tts", 0.5)  # Provide defaults
        else:
            # Use first available glass type as fallback
            first_glass = list(st.session_state.glass_props_session[position].keys())[0]
            props = st.session_state.glass_props_session[position][first_glass]
            return props.get("Te", 0.5), props.get("Tts", 0.5)
    except (KeyError, IndexError):
        st.error(f"Error accessing glass properties for {position}")
        return 0.5, 0.5  # Safe defaults

# Get glass properties
Te_ws, Tts_ws = glass_selector("Windshield")
Te_sl, Tts_sl = glass_selector("Sidelite")
Te_bl, Tts_bl = glass_selector("Backlite")
Te_roof, Tts_roof = glass_selector("Roof")

# ---------------- Prediction ----------------
try:
    input_row = pd.DataFrame([{
        "SolarFlux": weather["SolarFlux"],
        "Temperature": weather["Temperature"],  # Fixed key name
        "WindSpeed": weather["WindSpeed"],
        "CloudCoverage": weather["CloudCoverage"],
        "Humidity": weather["Humidity"],
        "Te(WS)": Te_ws, "Tts(WS)": Tts_ws,
        "Te(SL)": Te_sl, "Tts(SL)": Tts_sl,
        "Te(BL)": Te_bl, "Tts(BL)": Tts_bl,
        "Te(roof)": Te_roof, "Tts(roof)": Tts_roof
    }])

    prediction = model.predict(input_row)[0]
    pred_value = round(prediction, 2)

    # Add to history
    st.session_state.pred_history.append(pred_value)

    # ---------------- Display ----------------
    st.success(f"🌡️ Latest Predicted Cabin Temperature: {pred_value} °C")

except KeyError as e:
    st.error(f"KeyError: Missing key {e}. Please check your data inputs.")
    st.write("Available weather keys:", list(weather.keys()))
except Exception as e:
    st.error(f"Error making prediction: {str(e)}")

# ---------------- History Section ----------------
st.subheader("📦 Previous Predictions")
if st.button("🗑️ Clear History (Keep Latest)"):
    if st.session_state.pred_history:
        st.session_state.pred_history = [st.session_state.pred_history[-1]]
        st.success("Old history cleared, latest kept!")

if len(st.session_state.pred_history) > 1:
    prev_values = st.session_state.pred_history[:-1]
    st.text("\n".join([f"{i+1}. {val} °C" for i, val in enumerate(prev_values)]))
else:
    st.write("No previous predictions yet.")
