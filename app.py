import json
import pickle

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Appliances Energy Prediction", page_icon="⚡", layout="wide")

# ---- Model location on Hugging Face (change here if it ever moves) ----
HF_REPO_ID = "AbWbSid/appliance-energy-model"
HF_MODEL_FILENAME = "energy_model_full_dataset.pkl"


def _get_hf_token():
    """Optional token for a PRIVATE repo, read from Streamlit Secrets (HF_TOKEN).
    The repo is public, so normally this returns None and no token is used."""
    try:
        return st.secrets.get("HF_TOKEN") or None
    except Exception:
        return None


@st.cache_resource(show_spinner="Downloading the model from Hugging Face (first run only)...")
def load_model():
    # hf_hub_download stores the file in the local Hugging Face cache and reuses it afterwards.
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        token=_get_hf_token(),
    )
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


try:
    info = load_info()
except FileNotFoundError:
    st.error("`model_info.json` was not found. It must be in the same GitHub folder as `app.py`.")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"`model_info.json` is not valid JSON: {e}")
    st.stop()

try:
    model = load_model()
except Exception as e:
    name = type(e).__name__
    if name in ("RepositoryNotFoundError", "GatedRepoError"):
        st.error(
            f"Could not access the Hugging Face repository `{HF_REPO_ID}`. Check that it exists and "
            "is public. If it is private, add `HF_TOKEN` to the app's Streamlit Secrets."
        )
    elif name in ("EntryNotFoundError", "RemoteEntryNotFoundError"):
        st.error(f"The file `{HF_MODEL_FILENAME}` was not found in `{HF_REPO_ID}`. Check the exact filename.")
    elif name in ("LocalEntryNotFoundError", "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout"):
        st.error("Could not reach Hugging Face to download the model. Check the connection and reload the page.")
    elif isinstance(e, (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError, EOFError)):
        st.error(
            "The model file was downloaded but could not be loaded. This is usually a scikit-learn "
            "version mismatch: install the same version that was used to train and save the model "
            "(see `requirements.txt`)."
        )
    else:
        st.error(f"Unexpected error while loading the model ({name}).")
    with st.expander("Technical details"):
        st.code(f"{name}: {e}")
    st.stop()

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
    try:
        prediction = float(model.predict(feature_vector)[0])
    except Exception as e:
        st.error(
            "The model could not make a prediction. The input columns may not match what the model "
            "was trained on, or the scikit-learn version may differ."
        )
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")
        st.stop()
    prediction = max(prediction, 0.0)

    a, b = st.columns(2)
    a.metric("Predicted appliances energy (next 10 min)", f"{prediction:,.1f} Wh")
    b.metric("Average power during the interval", f"{prediction * 6 / 1000:,.2f} kW")
    st.info(f"Typical error of this model: about ±{info['cv_rmse']:,.0f} Wh (cross-validation RMSE).")
