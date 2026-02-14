import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os 
import google.generativeai as genai

import shap
import joblib
import streamlit as st 
from func import get_top_drivers , classify_risk , enrich_driver_info

import gdown 


MODEL_PATH = "shap_explainer.pkl"
FILE_ID = "1uzE6bVImoe49nsiLbQ8X_1SGz9XaJaW0"

@st.cache_resource
def load_explainer():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    return joblib.load(MODEL_PATH)

explainer = load_explainer()

st.success("Explainer Loaded Successfully")


model_xgb = joblib.load('model_xgb.pkl')
transformer = joblib.load('transformer.pkl')

st.title('Silica Impurity Predictor and Optmizer System')
st.header(':orange[This application will predict Silica Impurity and tuning assitance for reagants & airflow]')




iron_feed = st.number_input(
    "% Iron Feed",
    value=56.294739
)

silica_feed = st.number_input(
    "% Silica Feed",
    value=14.651716
)



starch_flow = st.number_input(
    "Starch Flow (m3/h)",
    value=2869.140569
)

amina_flow = st.number_input(
    "Amina Flow (m3/h)",
    value=488.144697
)



ore_pulp_flow = st.number_input(
    "Ore Pulp Flow (t/h)",
    value=397.578372
)

ore_pulp_ph = st.number_input(
    "Ore Pulp pH",
    min_value=0.0,
    max_value=14.0,
    value=9.767639
)

ore_pulp_density = st.number_input(
    "Ore Pulp Density (kg/cm³)",
    value=1.680380
)


air_flow_f1 = st.number_input("Airflow Flotation column 1", value=280.151856)
air_flow_f2 = st.number_input("Airflow Flotation column 2", value=277.159965)
air_flow_f3 = st.number_input("Airflow Flotation column 3", value=281.082397)
air_flow_f4 = st.number_input("Airflow Flotation column 4", value=299.447794)
air_flow_f5 = st.number_input("Airflow Flotation column 5", value=299.917814)
air_flow_f6 = st.number_input("Airflow Flotation column 6", value=292.071485)
air_flow_f7 = st.number_input("Airflow Flotation column 7", value=290.754856)



froth_lvl_f1 = st.number_input("Froth Flotation column 1", value=520.244823)
froth_lvl_f2 = st.number_input("Froth Flotation column 2", value=522.649555)
froth_lvl_f3 = st.number_input("Froth Flotation column 3", value=531.352662)
froth_lvl_f4 = st.number_input("Froth Flotation column 4", value=420.320973)
froth_lvl_f5 = st.number_input("Froth Flotation column 5", value=425.251706)
froth_lvl_f6 = st.number_input("Froth Flotation column 6", value=429.941018)
froth_lvl_f7 = st.number_input("Froth Flotation column 7", value=421.021231)



if st.button("Submit"):

    data = pd.DataFrame([[
        iron_feed, silica_feed,
        starch_flow, amina_flow,
        ore_pulp_flow, ore_pulp_ph, ore_pulp_density,
        air_flow_f1, air_flow_f2, air_flow_f3, air_flow_f4, air_flow_f5, air_flow_f6, air_flow_f7,
        froth_lvl_f1, froth_lvl_f2, froth_lvl_f3, froth_lvl_f4, froth_lvl_f5, froth_lvl_f6, froth_lvl_f7
    ]], columns=[
        "% Iron Feed", "% Silica Feed",
        "Starch Flow", "Amina Flow",
        "Ore Pulp Flow", "Ore Pulp pH", "Ore Pulp Density",
        "Flotation Column 01 Air Flow",
        "Flotation Column 02 Air Flow",
        "Flotation Column 03 Air Flow",
        "Flotation Column 04 Air Flow",
        "Flotation Column 05 Air Flow",
        "Flotation Column 06 Air Flow",
        "Flotation Column 07 Air Flow",
        "Flotation Column 01 Level",
        "Flotation Column 02 Level",
        "Flotation Column 03 Level",
        "Flotation Column 04 Level",
        "Flotation Column 05 Level",
        "Flotation Column 06 Level",
        "Flotation Column 07 Level"
    ])
    transformed_data = transformer.transform(data)
    transformed_data = pd.DataFrame(transformed_data, columns=data.columns)
    st.success("Inputs recorded successfully!")
    st.write({
        "% Iron Feed": iron_feed,
        "% Silica Feed": silica_feed,
        "Starch Flow (m3/h)": starch_flow,
        "Amina Flow (m3/h)": amina_flow,
        "Ore Pulp Flow (t/h)": ore_pulp_flow,
        "Ore Pulp pH": ore_pulp_ph,
        "Ore Pulp Density (kg/cm³)": ore_pulp_density,
        "Airflow F1": air_flow_f1,
        "Airflow F2": air_flow_f2,
        "Airflow F3": air_flow_f3,
        "Airflow F4": air_flow_f4,
        "Airflow F5": air_flow_f5,
        "Airflow F6": air_flow_f6,
        "Airflow F7": air_flow_f7,
        "Froth Level F1": froth_lvl_f1,
        "Froth Level F2": froth_lvl_f2,
        "Froth Level F3": froth_lvl_f3,
        "Froth Level F4": froth_lvl_f4,
        "Froth Level F5": froth_lvl_f5,
        "Froth Level F6": froth_lvl_f6,
        "Froth Level F7": froth_lvl_f7
      })





  






    pred = model_xgb.predict(transformed_data.iloc[0:1])

    st.write(pred)

    risk_level = classify_risk(pred)

    st.write(risk_level)



    shap_values = explainer(transformed_data)


    feature_name = transformed_data.columns.tolist()

    drivers = get_top_drivers(shap_values[0],feature_names=feature_name)

    top_features = enrich_driver_info(drivers, row_data=transformed_data.iloc[0:1])

    prompt = f'''You are a flotation process decision-support assistant have in depth knowledge in reverse flotation of iron mining. 

    Predicted Silica Level: {pred}
    Risk Level: {risk_level}
    

    Key Contributing Factors:
    {top_features}

    measurement metrics
    Starch (reagent) Flow measured in m3/h
    Amina (reagent) Flow measured in m3/h
    Ore Pulp Flow Rate - t/h
    Ore Pulp pH Level- pH scale from 0 to 14
    Ore Pulp Density- Density scale from 1 to 3 kg/cm³
    Air flow that goes into the flotation column measured in Nm³/h
    Froth level in the flotation column measured in mm



    Generate a concise operational recommendation to achieve lower silica impurity.
    Guidelines:
    - Suggest cautious, incremental adjustments only.(provide numerical range with increase or reduce direction)
    - Do NOT recommend extreme parameter changes.
    - Do NOT claim causal certainty.
    - Frame guidance as advisory for human operators.
    - Keep response under 200 words.
    - do not suggest any change in silica fee and iron feed as it is out of control
    - a tabular format what, how much and direction to change  '''

    gemini_key = os.getenv('Google_API_KEY1')
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    response = model.generate_content(prompt)


    st.write(response.text)