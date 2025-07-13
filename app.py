import streamlit as st
import pandas as pd
import numpy as np
import joblib
import copy

# Load trained model (no preprocessor)
model = joblib.load('ThermalComfort_prediction_model.pkl')

# Weather data for each city
city_weather = {
    "Mumbai": {"Tempearture": 31, "SolarFlux": 932, "Humidity": 65, "WindSpeed": 4, "CloudCoverage": 5},
    "Jodhpur": {"Tempearture": 40, "SolarFlux": 925, "Humidity": 25, "WindSpeed": 1, "CloudCoverage": 5},
    "Bengaluru": {"Tempearture": 28, "SolarFlux": 946, "Humidity": 54, "WindSpeed": 3, "CloudCoverage": 3}
}

# Glass properties
glass_props = {
    "Windshield": {
        "TSANx//TSANx": {"Te": 0.536, "t": 5.0, "Tts": 0.647},
        "TSA3+//TSA3+": {"Te": 0.425, "t": 5.0, "Tts": 0.568},
        "TSANx//TSA3+": {"Te": 0.449, "t": 5.0, "Tts": 0.585}},
    "Sidelite": {
        "TSANx": {"Te": 0.634, "t": 3.2, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "t": 3.2, "Tts": 0.619},
        "Inshade": {"Te": 0.401, "t": 3.2, "Tts": 0.521}},
    "Backlite": {
        "TSANx": {"Te": 0.634, "t": 3.2, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "t": 3.2, "Tts": 0.619},
        "TSA5+": {"Te": 0.360, "t": 3.2, "Tts": 0.522}},
    "Roof": {
        "VG10": {"Te": 0.1, "t":3.85, "Tts": 0.336},
        "VG20": {"Te": 0.165, "t":3.85, "Tts": 0.383}}
}

if "glass_props_session" not in st.session_state:
    st.session_state.glass_props_session = copy.deepcopy(glass_props)

st.set_page_config(page_title="Thermal comfort Predictor", layout="centered")
st.title("🚗 Thermal Comfort Dashboard")

# Initialize session state dict
if "city_weather_session" not in st.session_state:
    st.session_state.city_weather_session = copy.deepcopy(city_weather)

# Build city list from session_state dict keys
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
                "Tempearture": temp,
                "SolarFlux": solar,
                "Humidity": humidity,
                "WindSpeed": wind,
                "CloudCoverage": cloud
            }
            st.success(f"City '{new_city_name}' added.")
            # Optionally reset the selectbox to new city
            st.experimental_rerun()

else:
    weather = st.session_state.city_weather_session[city]
    st.write(f"Selected city weather data: {weather}")

# Glass selector function
def glass_selector(position):
    glass_list = list(st.session_state.glass_props_session[position].keys()) + ["➕ Add New Glass Type"]
    selected = st.selectbox(f"{position} Glass Type", glass_list, key=position)

    if selected == "➕ Add New Glass Type":
        with st.expander(f"Add New Glass Type to {position}"):
            new_name = st.text_input("Glass Name", key=f"{position}_name")
            new_trans = st.slider("Transmittance", 0.2, 0.9, 0.6, key=f"{position}_Te")
            new_thick = st.slider("Thickness (mm)", 3.0, 6.0, 4.5, key=f"{position}_t")
            new_Tts = st.slider("Transmitted Solar Energy", 0.0, 0.7, 0.3, key=f"{position}_Tts")

            if st.button("Add", key=f"{position}_add"):
                if new_name and new_name not in st.session_state.glass_props_session[position]:
                    st.session_state.glass_props_session[position][new_name] = {
                        "Te": new_trans,
                        "t": new_thick,
                        "Tts": new_Tts
                    }
                    st.success(f"✅ Added '{new_name}' to {position}")
                else:
                    st.warning("Name exists or is empty. Please enter a unique name.")

    # Return properties for the selected glass (if exists)
    props = st.session_state.glass_props_session[position].get(selected, None)
    if props is None:
        # fallback to first glass type to avoid KeyError
        fallback = list(st.session_state.glass_props_session[position].values())[0]
        return fallback["t"], fallback["Te"], fallback["Tts"]
    return props["t"], props["Te"], props["Tts"]

t_ws, Te_ws, Tts_ws = glass_selector("Windshield")
t_sl, Te_sl, Tts_sl = glass_selector("Sidelite")
t_bl, Te_bl, Tts_bl = glass_selector("Backlite")
t_roof, Te_roof, Tts_roof = glass_selector("Roof")

# Build input row
input_row = pd.DataFrame([{
    "SolarFlux": weather["SolarFlux"],
    "Tempearture": weather["Tempearture"],
    "WindSpeed": weather["WindSpeed"],
    "CloudCoverage": weather["CloudCoverage"],
    "Humidity": weather["Humidity"],
    "t(WS)": t_ws, "Te(WS)": Te_ws, "Tts(WS)": Tts_ws,
    "t(SL)": t_sl, "Te(SL)": Te_sl, "Tts(SL)": Tts_sl,
    "t(BL)": t_bl, "Te(BL)": Te_bl, "Tts(BL)": Tts_bl,
    "t(roof)": t_roof, "Te(roof)": Te_roof, "Tts(roof)": Tts_roof
}])

# Predict and export
if st.button("🔍 Predict Cabin Temperature"):
    prediction = model.predict(input_row)[0]
    st.success(f"🌡️ Predicted Cabin Temperature: **{prediction:.2f} °C**")
