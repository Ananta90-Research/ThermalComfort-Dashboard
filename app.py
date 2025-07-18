import streamlit as st
import pandas as pd
import numpy as np
import joblib
import copy
import base64
from PIL import Image
from st_clickable_images import clickable_images
import plotly.graph_objects as go

# --- Load model ---
model = joblib.load('ThermalComfort_prediction_model.pkl')

# --- Default data ---
city_weather = {
    "Mumbai": {"Tempearture":31,"SolarFlux":932,"Humidity":65,"WindSpeed":4,"CloudCoverage":5},
    "Jodhpur": {"Tempearture":40,"SolarFlux":925,"Humidity":25,"WindSpeed":1,"CloudCoverage":5},
    "Bengaluru": {"Tempearture":28,"SolarFlux":946,"Humidity":54,"WindSpeed":3,"CloudCoverage":3}
}
glass_props = {
    "Windshield": {
        "TSANx//TSANx": {"Te":0.536,"t":5.0,"Tts":0.647},
        "TSA3+//TSA3+": {"Te":0.425,"t":5.0,"Tts":0.568},
        "TSANx//TSA3+": {"Te":0.449,"t":5.0,"Tts":0.585}
    },
    "Sidelite": {
        "TSANx": {"Te":0.634,"t":3.2,"Tts":0.720},
        "TSA3+": {"Te":0.496,"t":3.2,"Tts":0.619},
        "Inshade": {"Te":0.401,"t":3.2,"Tts":0.521}
    },
    "Backlite": {
        "TSANx": {"Te":0.634,"t":3.2,"Tts":0.720},
        "TSA3+": {"Te":0.496,"t":3.2,"Tts":0.619},
        "TSA5+": {"Te":0.360,"t":3.2,"Tts":0.522}
    },
    "Rear Windshield": {
        "TSANx": {"Te":0.634,"t":3.2,"Tts":0.720},
        "TSA3+": {"Te":0.496,"t":3.2,"Tts":0.619},
        "TSA5+": {"Te":0.360,"t":3.2,"Tts":0.522}
    },
    "Roof": {
        "VG10": {"Te":0.1,"t":3.85,"Tts":0.336},
        "VG20": {"Te":0.165,"t":3.85,"Tts":0.383}
    }
}
glass_positions = list(glass_props.keys())

# --- Session state setup ---
if "city_weather_session" not in st.session_state:
    st.session_state.city_weather_session = copy.deepcopy(city_weather)
if "glass_props_session" not in st.session_state:
    st.session_state.glass_props_session = copy.deepcopy(glass_props)

# --- Load clickable SUV image ---
@st.cache_data
def load_img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_b64 = load_img_b64("suv_top_view.png")  # Your SUV image with transparent zones

# --- Page config ---
st.set_page_config(page_title="Thermal Comfort Dashboard", layout="wide")
st.title("🚗 Thermal Comfort Dashboard")

# --- Show SUV image with clickable zones ---
col_a, col_b, col_c = st.columns([1,3,1])
with col_b:
    clicked = clickable_images(
        [f"data:image/png;base64,{img_b64}"],
        titles=glass_positions,
        div_style={"display":"flex","justify-content":"center"},
        img_style={"height":"400px"}
    )

# --- City add/delete UI ---
city_opts = list(st.session_state.city_weather_session.keys()) + ["➕ Add City", "🗑️ Delete City"]
city_sel = st.sidebar.selectbox("City / Weather Action", city_opts)

if city_sel == "➕ Add City":
    nc = st.sidebar.text_input("New City")
    t = st.sidebar.slider("Temperature (°C)",20,50,35)
    sf = st.sidebar.slider("Solar Flux",500,1200,900)
    hum = st.sidebar.slider("Humidity (%)",10,100,50)
    ws = st.sidebar.slider("Wind Speed (km/h)",0,30,10)
    cc = st.sidebar.slider("Cloud Coverage (%)",0,100,20)
    if st.sidebar.button("Add City"):
        if nc and nc not in st.session_state.city_weather_session:
            st.session_state.city_weather_session[nc] = {"Tempearture":t,"SolarFlux":sf,"Humidity":hum,"WindSpeed":ws,"CloudCoverage":cc}
            st.success(f"Added city {nc}")
            st.rerun()
        else:
            st.sidebar.warning("Enter a unique name.")
elif city_sel == "🗑️ Delete City":
    defaults = set(city_weather.keys())
    customs = [c for c in st.session_state.city_weather_session if c not in defaults]
    if customs:
        dc = st.sidebar.selectbox("Pick city to delete", customs)
        if st.sidebar.button("Delete City"):
            st.session_state.city_weather_session.pop(dc)
            st.success(f"Deleted {dc}")
            st.rerun()
    else:
        st.sidebar.info("No custom cities")

# Ensure a valid city is selected
if city_sel in st.session_state.city_weather_session:
    weather = st.session_state.city_weather_session[city_sel]
else:
    weather = next(iter(st.session_state.city_weather_session.values()))

# --- Glass selector function ---
def glass_selector(position):
    session = st.session_state.glass_props_session[position]
    default_keys = set(glass_props[position].keys())
    opts = list(session.keys()) + ["➕ Add New"]
    sel = st.selectbox(f"{position} Glass", opts, key=position)
    # Delete button
    if sel != "➕ Add New" and sel not in default_keys:
        if st.button(f"🗑️ Delete {position}", key=f"del_{position}"):
            session.pop(sel)
            st.success(f"Deleted {sel}")
            st.rerun()
    # Add new
    if sel == "➕ Add New":
        with st.expander(f"Add glass for {position}"):
            n = st.text_input("Name",key=f"{position}_n")
            te = st.slider("Transmittance",0.2,0.9,0.5,key=f"{position}_te")
            t = st.slider("Thickness",3.0,6.0,4.5,key=f"{position}_t")
            tts = st.slider("Trans Sol Energy",0.0,0.7,0.3,key=f"{position}_tts")
            if st.button("Add", key=f"add_{position}"):
                if n and n not in session:
                    session[n] = {"Te":te,"t":t,"Tts":tts}
                    st.success(f"Added {n}")
                    st.rerun()
                else:
                    st.warning("Enter unique name")
    props = session.get(sel) or next(iter(session.values()))
    return props["t"], props["Te"], props["Tts"]

# --- Gather glass params based on click ---
t_params = {}
for i,pos in enumerate(glass_positions):
    if clicked == i:
        t_params[pos] = glass_selector(pos)
    else:
        p = next(iter(st.session_state.glass_props_session[pos].values()))
        t_params[pos] = (p["t"],p["Te"],p["Tts"])

# Unpack
t_ws,Te_ws,Tts_ws = t_params["Windshield"]
t_sl,Te_sl,Tts_sl = t_params["Sidelite"]
t_bl,Te_bl,Tts_bl = t_params["Backlite"]
t_rw,Te_rw,Tts_rw = t_params["Rear Windshield"]
t_rf,Te_rf,Tts_rf = t_params["Roof"]

# --- Prediction on button ---
if st.sidebar.button("🔍 Predict Cabin Temp"):
    df = pd.DataFrame([{
        "SolarFlux": weather["SolarFlux"],
        "Tempearture": weather["Tempearture"],
        "WindSpeed": weather["WindSpeed"],
        "CloudCoverage": weather["CloudCoverage"],
        "Humidity": weather["Humidity"],
        "t(WS)":t_ws,"Te(WS)":Te_ws,"Tts(WS)":Tts_ws,
        "t(SL)":t_sl,"Te(SL)":Te_sl,"Tts(SL)":Tts_sl,
        "t(BL)":t_bl,"Te(BL)":Te_bl,"Tts(BL)":Tts_bl,
        "t(RW)":t_rw,"Te(RW)":Te_rw,"Tts(RW)":Tts_rw,
        "t(roof)":t_rf,"Te(roof)":Te_rf,"Tts(roof)":Tts_rf
    }])
    pred = model.predict(df)[0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        gauge={'axis':{'range':[0,50]},
               'steps':[{'range':[0,25],'color':'lightgreen'},
                        {'range':[25,35],'color':'yellow'},
                        {'range':[35,50],'color':'red'}],
               'threshold':{'line':{'color':'darkred','width':4},'value':40}},
        title={'text':"Predicted Cabin Temp (°C)"}
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.sidebar.metric("Cabin Temp", f"{pred:.2f} °C")
