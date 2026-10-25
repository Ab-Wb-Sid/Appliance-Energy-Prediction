import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Appliances Energy Prediction", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model():
    with open("energy_model_full_dataset.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


model = load_model()
info = load_info()
stats = info["stats"]

st.title("⚡ Appliances Energy Prediction System")
st.write(
    "Predict how much energy (Wh) the appliances in a house will use in the next 10 minutes, "
    "from the indoor sensors, the weather outside and the time of day."
)
st.caption(f"Model: {info['best_algorithm']}  |  Cross-validation RMSE: about {info['cv_rmse']:,.1f} Wh")


def number_box(feature, label, step=0.1, fmt="%.1f"):
    s = stats[feature]
    return st.number_input(label, min_value=float(s["min"]), max_value=float(s["max"]),
                           value=float(s["median"]), step=step, format=fmt, key=feature)


values = {}

st.subheader("1. Time and lights")
c1, c2, c3 = st.columns(3)
with c1:
    values["Hour"] = st.slider("Hour of the day (0-23)", 0, 23, 18)
with c2:
    values["Day of Week"] = st.selectbox("Day of the week", info["days"])
with c3:
    values["lights"] = number_box("lights", "Lights energy (Wh)", step=10.0, fmt="%.0f")

st.subheader("2. Indoor temperature (°C)")
cols = st.columns(3)
for i in range(1, 10):
    with cols[(i - 1) % 3]:
        values[f"T{i}"] = number_box(f"T{i}", f"T{i}")

st.subheader("3. Indoor humidity (%)")
cols = st.columns(3)
for i in range(1, 10):
    with cols[(i - 1) % 3]:
        values[f"RH_{i}"] = number_box(f"RH_{i}", f"RH_{i}")

st.subheader("4. Weather outside")
c1, c2, c3 = st.columns(3)
with c1:
    values["T_out"] = number_box("T_out", "Outside temperature (°C)")
    values["Windspeed"] = number_box("Windspeed", "Wind speed (m/s)")
with c2:
    values["RH_out"] = number_box("RH_out", "Outside humidity (%)")
    values["Visibility"] = number_box("Visibility", "Visibility (km)")
with c3:
    values["Press_mm_hg"] = number_box("Press_mm_hg", "Pressure (mm Hg)")
    values["Tdewpoint"] = number_box("Tdewpoint", "Dew point (°C)")

if st.button("Predict Energy Use", type="primary"):
    feature_vector = pd.DataFrame([values])[info["numeric_features"] + info["categorical_features"]]
    prediction = float(model.predict(feature_vector)[0])
    prediction = max(prediction, 0.0)

    a, b = st.columns(2)
    a.metric("Predicted appliances energy (next 10 min)", f"{prediction:,.1f} Wh")
    b.metric("Average power during the interval", f"{prediction * 6 / 1000:,.2f} kW")
    st.info(f"Typical error of this model: about ±{info['cv_rmse']:,.0f} Wh (cross-validation RMSE).")
