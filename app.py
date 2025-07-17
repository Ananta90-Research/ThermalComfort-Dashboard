import streamlit as st
import pandas as pd
import numpy as np
import joblib
import copy

# Load model
model = joblib.load('ThermalComfort_prediction_model.pkl')

# Default data
city_weather = {
    "Mumbai": {"Tempearture": 31, "SolarFlux": 932, "Humidity": 65, "WindSpeed": 4, "CloudCoverage": 5},
    "Jodhpur": {"Tempearture": 40, "SolarFlux": 925, "Humidity": 25, "WindSpeed": 1, "CloudCoverage": 5},
    "Bengaluru": {"Tempearture": 28, "SolarFlux": 946, "Humidity": 54, "WindSpeed": 3, "CloudCoverage": 3}
}
glass_props = {
    "Windshield": {
        "TSANx//TSANx": {"Te": 0.536, "t": 5.0, "Tts": 0.647},
        "TSA3+//TSA3+": {"Te": 0.425, "t": 5.0, "Tts": 0.568},
        "TSANx//TSA3+": {"Te": 0.449, "t": 5.0, "Tts": 0.585}
    },
    "Sidelite": {
        "TSANx": {"Te": 0.634, "t": 3.2, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "t": 3.2, "Tts": 0.619},
        "Inshade": {"Te": 0.401, "t": 3.2, "Tts": 0.521}
    },
    "Backlite": {
        "TSANx": {"Te": 0.634, "t": 3.2, "Tts": 0.720},
        "TSA3+": {"Te": 0.496, "t": 3.2, "Tts": 0.619},
        "TSA5+": {"Te": 0.360, "t": 3.2, "Tts": 0.522}
    },
    "Roof": {
        "VG10": {"Te": 0.1, "t": 3.85, "Tts": 0.336},
        "VG20": {"Te": 0.165, "t": 3.85, "Tts": 0.383}
    }
}

# Session state init
if "city_weather_session" not in st.session_state:
    st.session_state.city_weather_session = copy.deepcopy(city_weather)
if "glass_props_session" not in st.session_state:
    st.session_state.glass_props_session = copy.deepcopy(glass_props)

st.set_page_config(page_title="Thermal Comfort Predictor", layout="centered")
st.title("🚗 Thermal Comfort Dashboard")

# Callback definitions
def delete_city():
    ct = st.session_state.delete_city
    del st.session_state.city_weather_session[ct]
    st.experimental_rerun()

def make_delete_glass_cb(position):
    def cb():
        name = st.session_state[f"sel_{position}"]
        del st.session_state.glass_props_session[position][name]
        # Clear selectbox key
        del st.session_state[f"sel_{position}"]
        st.experimental_rerun()
    return cb

# City selector with add/delete
city_opts = list(st.session_state.city_weather_session.keys()) + ["➕ Add City","🗑️ Delete City"]
city = st.selectbox("Select City or Action", city_opts)

if city == "➕ Add City":
    new_city = st.text_input("City Name")
    temp = st.slider("Temperature (°C)", 20, 50, 35)
    solar = st.slider("Solar Flux (W/m²)", 500, 1200, 900)
    humidity = st.slider("Humidity (%)", 10, 100, 50)
    wind = st.slider("Wind Speed (km/h)", 0, 30, 10)
    cloud = st.slider("Cloud Coverage (%)", 0, 100, 20)
    if st.button("Add City"):
        if new_city and new_city not in st.session_state.city_weather_session:
            st.session_state.city_weather_session[new_city] = {
                "Tempearture": temp, "SolarFlux": solar,
                "Humidity": humidity, "WindSpeed": wind, "CloudCoverage": cloud
            }
            st.success(f"City '{new_city}' added.")
            st.experimental_rerun()
        else:
            st.warning("Enter a unique city name.")
elif city == "🗑️ Delete City":
    defaults = set(city_weather.keys())
    customs = [c for c in st.session_state.city_weather_session if c not in defaults]
    if customs:
        st.selectbox("Delete city", customs, key="delete_city")
        st.button("Delete", on_click=delete_city)
    else:
        st.info("No custom cities to delete.")
    st.stop()
else:
    weather = st.session_state.city_weather_session[city]

# Inline glass selector with add & delete
def glass_selector(position):
    session = st.session_state.glass_props_session[position]
    default_set = set(glass_props[position].keys())
    opts = list(session.keys()) + ["➕ Add New"]
    selected = st.selectbox(f"{position} glass", opts, key=f"sel_{position}")

    # Show delete button only for custom
    if selected != "➕ Add New" and selected not in default_set:
        st.button("🗑️ Delete this glass", on_click=make_delete_glass_cb(position), key=f"btn_del_{position}")

    # Add new type
    if selected == "➕ Add New":
        with st.expander(f"Add new {position} glass"):
            n = st.text_input("Name", key=f"{position}_n")
            te = st.slider("Transmittance", 0.2, 0.9, 0.6, key=f"{position}_te")
            t = st.slider("Thickness (mm)", 3.0, 6.0, 4.5, key=f"{position}_t")
            tts = st.slider("Transmitted Solar Energy", 0.0, 0.7, 0.3, key=f"{position}_tts")
            if st.button("Add Glass", key=f"add_{position}"):
                if n and n not in session:
                    session[n] = {"Te": te, "t": t, "Tts": tts}
                    st.success(f"Glass '{n}' added.")
                else:
                    st.warning("Enter a unique name.")

    # Fetch properties (fallback to any valid)
    props = session.get(selected) if selected != "➕ Add New" else None
    if props is None:
        props = next(iter(session.values()))

    return props["t"], props["Te"], props["Tts"]

# Collect glass data
t_ws, Te_ws, Tts_ws = glass_selector("Windshield")
t_sl, Te_sl, Tts_sl = glass_selector("Sidelite")
t_bl, Te_bl, Tts_bl = glass_selector("Backlite")
t_rf, Te_rf, Tts_rf = glass_selector("Roof")

# Prediction action
if st.button("🔍 Predict Cabin Temperature"):
    df = pd.DataFrame([{
        "SolarFlux": weather["SolarFlux"],
        "Tempearture": weather["Tempearture"],
        "WindSpeed": weather["WindSpeed"],
        "CloudCoverage": weather["CloudCoverage"],
        "Humidity": weather["Humidity"],
        "t(WS)": t_ws, "Te(WS)": Te_ws, "Tts(WS)": Tts_ws,
        "t(SL)": t_sl, "Te(SL)": Te_sl, "Tts(SL)": Tts_sl,
        "t(BL)": t_bl, "Te(BL)": Te_bl, "Tts(BL)": Tts_bl,
        "t(roof)": t_rf, "Te(roof)": Te_rf, "Tts(roof)": Tts_rf
    }])
    pred = model.predict(df)[0]
    st.success(f"🌡️ Predicted Cabin Temperature: **{pred:.2f} °C**")
