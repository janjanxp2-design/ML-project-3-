import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64

st.set_page_config(page_title='Hotel booking Ml Predictor',page_icon = '🏨' , layout="centered")
aircraft="MODEL_FILES"
st.title("Dynmic Hotel booking ML predictor")
st.markdown("This is a ML predector of Hotel booking status used calssification and regression")

# back groun video 
def set_video_bg(video_file):
    # 1. ویڈیو فائل کو پڑھنا
    with open(video_file, "rb") as f:
        video_bytes = f.read()

    # 2. Base64 میں تبدیل کرنا
    b64_video = base64.b64encode(video_bytes).decode()

    # 3. HTML اور CSS کے ذریعے ویڈیو کو لائیو پس منظر میں لگانا
    video_html = f"""
    <style>
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
        filter: brightness(0.6); /* تاکہ تحریر صاف نظر آئے */
    }}
    .stApp {{
        background: transparent;
    }}
    </style>
    <video autoplay loop muted playsinline id="bg-video">
        <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)


set_video_bg("bg_video.mp4")
# 1 discover available models
available = (f.replace('_schema.json','') for f in os.listdir(aircraft) if f.endswith ('_schema.json'))
if not available:
    st.error("NO model in aircraf pls reload")
    st.stop
choice=st.selectbox("Choose a model to use",available)

# 2 load choosen model 
@st.cache_resource
def load_model(name):
    model=joblib.load(f'{aircraft}/{name}_model.pkl')
    scaler_path=f"{aircraft}/{name}_scaler.pkl"
    scaler=joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    with open(f"{aircraft}/{name}_schema.json") as f:
        schema = json.load(f)
    return model,scaler,schema
model , scaler , schema = load_model(choice)
st.subheader(f"Task: {schema['task_type'].title()}  |  Model: {schema['model_name']}")

# 3 Dynamically build the input from schema
st.markdown("## Enter Feature File")
user_input={}
with st.form("pridictor_form"):
    cols = st.columns(2)
    for i ,feat in enumerate(schema['features']):
        target_col = cols[i % 2]
        if feat['dtype'] == "numeric":
            user_input[feat['name']] = target_col.slider(
                feat['name'],
                min_value=float(0),
                max_value=float(feat["max"]),
                step=float(feat['step'])
            )
        elif feat['dtype'] == 'categorical':
            user_input[feat['name']] = target_col.selectbox(feat['name'], feat['options'])
    submitted=st.form_submit_button("Predict")

# 4 predict + display result (branches by task)
if submitted:
    feature_order = [f['name'] for f in schema['features']]
    input_df = pd.DataFrame([[user_input[f] for f in feature_order]], columns=feature_order)
    input_values = scaler.transform(input_df) if scaler is not None else input_df.values
    if schema['task_type'] == 'regression':
        prediction =  model.predict(input_values)[0]
        st.success(f"Predicted {schema['target_name']}:   {prediction:.3f}  ")
    elif schema['task_type'] == "classification":
        prediction = model.predict(input_values)[0]
        class_name = schema.get('class_names', [str(prediction)])
        prediction_lable = prediction
        st.success(f"Predicted {schema['target_name']}:   **  {prediction_lable}  ** ")
        if hasattr(model, "predict_proba"):
            proba =  model.predict_proba(input_values)[0]
            proba_df = pd.DataFrame({"Class" : class_name , "probability" : proba})
            st.bar_chart(proba_df.set_index('Class'))

st.markdown("___")
