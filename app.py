###########################################################
# 🩺 THYROID CANCER PREDICTOR — FAST MODE (NO SHAP)
# Stable for Streamlit Cloud — Low RAM, Instant Output
###########################################################

import streamlit as st
import pandas as pd
import numpy as np
import joblib, gdown, json

st.set_page_config(page_title="🩺 Thyroid Cancer Predictor", layout="wide")

############################################################
# 🔥 MODEL LOADING FROM GOOGLE DRIVE (CACHED)
############################################################

FILE_ID = "1OeXItzXaTEvZFgOg6FldYJjznF7p4D5Y"    # your 851MB PKL file
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    gdown.download(URL, "model.pkl", quiet=False)
    return joblib.load("model.pkl")

model = load_model()
st.success("Model Loaded Successfully (Google Drive Cached) :)")


############################################################
# LOAD FEATURE COLUMN ORDER
############################################################

with open("feature_columns.json") as f:
    feature_cols = json.load(f)


############################################################
# INPUT UI
############################################################

st.title("🩺 AI-Driven Thyroid Cancer Risk Assessor (FAST MODE)")

tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🧬 Medical History", "🧪 Lab Values"])
data = {}

# TAB 1 ----------------------------------
with tab1:
    st.subheader("👤 Demographic Indicators")
    col1,col2,col3 = st.columns(3)
    with col1: data["Age"] = st.number_input("Age",1,120,30,step=1)
    with col2: data["Gender"] = st.radio("Gender",["Male","Female"],horizontal=True)
    with col3: data["Country"] = st.radio("Country",["India","China","Nigeria","Russia","Brazil"],horizontal=True)

# TAB 2 ----------------------------------
with tab2:
    st.subheader("🧬 Family & Lifestyle Determinants")
    col4,col5,col6 = st.columns(3)
    yn=["No","Yes"]
    with col4:
        data["Family_History"]=st.radio("Family History of Thyroid Cancer",yn,horizontal=True)
        data["Smoking"]=st.radio("Smoking",yn,horizontal=True)
    with col5:
        data["Obesity"]=st.radio("Obesity",yn,horizontal=True)
        data["Diabetes"]=st.radio("Diabetes",yn,horizontal=True)
    with col6:
        data["Radiation_Exposure"]=st.radio("Radiation Exposure",yn,horizontal=True)
        data["Iodine_Deficiency"]=st.radio("Iodine Deficiency",yn,horizontal=True)

# TAB 3 ----------------------------------
with tab3:
    st.subheader("🧪 Clinical / Thyroid Function Parameters")
    colA,colB,colC,colD=st.columns(4)
    with colA: data["TSH_Level"]=st.number_input("TSH Level",step=0.01)
    with colB: data["T3_Level"]=st.number_input("T3 Level",step=0.01)
    with colC: data["T4_Level"]=st.number_input("T4 Level",step=0.01)
    with colD: data["Nodule_Size"]=st.number_input("Nodule Size (cm)",step=0.01)


############################################################
# DATA → ONE-HOT → MODEL FORMAT
############################################################

def prepare(df):
    df = pd.get_dummies(df)
    for c in feature_cols:
        if c not in df.columns:
            df[c]=0
    return df[feature_cols]

X = prepare(pd.DataFrame([data]))


############################################################
# RISK ENGINE (Fast Logic)
############################################################

def badge(prob):
    if prob < 0.33: return "🟢 LOW RISK (Likely Non-Cancerous)"
    if prob < 0.66: return "🟡 MODERATE RISK — Monitor Clinically"
    return "🔴 HIGH RISK — Suspicion of Malignancy"


def flags(d):
    out=[]
    if d["TSH_Level"]>4.5: out.append("⚠ High TSH → Possible Hypothyroidism")
    if d["T3_Level"]<0.8: out.append("⚠ Low T3 → Thyroid suppression")
    if d["T4_Level"]<4.5: out.append("⚠ Low T4 → Underfunction")
    if d["Nodule_Size"]>2: out.append("🚨 Nodule > 2cm — FNAC Recommended")
    return out


############################################################
# PREDICT
############################################################

if st.button("🔍 Predict Now"):
    y_prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    st.subheader("🩺 Diagnosis Result")

    if pred==0: st.success("Result: NON-CANCEROUS (0)")
    else: st.error("Result: CANCEROUS Suspicion (1)")

    st.write("### Risk Assessment")
    st.write(badge(y_prob))

    clinical = flags(data)
    if clinical:
        st.warning("### Clinical Alerts")
        for a in clinical: st.write(a)
    else:
        st.success("No critical biochemical or nodule risk detected")


st.caption("🏥 Thyroid Clinical-AI — Engineered by Group 5")
