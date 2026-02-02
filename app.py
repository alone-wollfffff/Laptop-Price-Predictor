import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)
# 2. Animated Background CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1, h2, h3, p, label {
        color: white !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Assets
@st.cache_resource
def load_assets():
    df = pickle.load(open('df.pkl', 'rb'))
    pipe = pickle.load(open('pipe_RandomForest.pkl', 'rb'))
    return df, pipe

df, pipe = load_assets()

# 4. Session State
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}

# --- STEP 0: WELCOME ---
if st.session_state.step == 0:
    st.title("💻 Laptop Price Predictor")
    st.write("Find the fair market value of any laptop configuration using AI.")
    if st.button("Start Prediction"):
        st.session_state.step = 1
        st.rerun()

# --- STEPS 1–5 ---
elif 1 <= st.session_state.step <= 5:
    st.progress(st.session_state.step / 5)

    if st.session_state.step == 1:
        st.subheader("Brand & Style")
        st.session_state.responses['company'] = st.selectbox(
            'Brand', sorted(df['Company'].unique()))
        st.session_state.responses['type'] = st.selectbox(
            'Category', sorted(df['TypeName'].unique()))

    elif st.session_state.step == 2:
        st.subheader("Performance")
        st.session_state.responses['cpu'] = st.selectbox(
            'Processor', sorted(df['Cpu Brand'].unique()))
        st.session_state.responses['ram'] = st.selectbox(
            'RAM (GB)', sorted(df['Ram'].unique()))
        st.session_state.responses['gpu'] = st.selectbox(
            'Graphics', sorted(df['Gpu brand'].unique()))

    elif st.session_state.step == 3:
        st.subheader("Storage")
        st.session_state.responses['ssd'] = st.selectbox(
            'SSD Size (GB)', [0, 128, 256, 512, 1024])
        st.session_state.responses['hdd'] = st.selectbox(
            'HDD Size (GB)', [0, 128, 256, 512, 1024])

    elif st.session_state.step == 4:
        st.subheader("Screen & Display Quality")

        standard_sizes = [11.6, 12.5, 13.3, 14.0, 15.6, 16.0, 17.3]

        st.session_state.responses['screen_size'] = st.select_slider(
            'Select Screen Size (inches)',
            options=standard_sizes,
            value=15.6
        )

        st.session_state.responses['res'] = st.selectbox(
            'Resolution',
            ['1920x1080', '1366x768', '1600x900', '3840x2160']
        )

        st.session_state.responses['touch'] = st.radio(
            'Touchscreen', ['No', 'Yes'], horizontal=True)
        st.session_state.responses['ips'] = st.radio(
            'IPS Panel', ['No', 'Yes'], horizontal=True)

    elif st.session_state.step == 5:
        st.subheader("Final Details")
        st.session_state.responses['weight'] = st.slider(
            'Weight (kg)', min_value=0.8, max_value=4.0, value=2.0, step=0.1)
        st.session_state.responses['os'] = st.selectbox(
            'Operating System', sorted(df['os'].unique()))

    # Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step -= 1
            st.rerun()

    with col2:
        if st.session_state.step < 5:
            if st.button("Next ➡️"):
                st.session_state.step += 1
                st.rerun()
        else:
            if st.button("🎯 Predict"):
                r = st.session_state.responses

                X_res = int(r['res'].split('x')[0])
                Y_res = int(r['res'].split('x')[1])
                ppi = ((X_res**2 + Y_res**2) ** 0.5) / r['screen_size']

                query = pd.DataFrame([[
                    r['company'],
                    r['type'],
                    r['ram'],
                    r['weight'],
                    1 if r['touch'] == 'Yes' else 0,
                    1 if r['ips'] == 'Yes' else 0,
                    ppi,
                    r['cpu'],
                    r['hdd'],
                    r['ssd'],
                    r['gpu'],
                    r['os']
                ]], columns=[
                    'Company', 'TypeName', 'Ram', 'Weight',
                    'Touchscreen', 'Ips', 'ppi',
                    'Cpu Brand', 'HDD', 'SSD',
                    'Gpu brand', 'os'
                ])

                pred_log = pipe.predict(query)[0]
                price = max(0, int(np.exp(pred_log)))

                st.session_state.prediction = price
                st.session_state.step = 6
                st.rerun()

# --- STEP 6: RESULT ---
elif st.session_state.step == 6:
    st.balloons()
    st.title("Estimated Market Price")
    st.header(f"₹{st.session_state.prediction:,}")
    if st.button("🔄 Restart Prediction"):
        st.session_state.step = 0

        st.rerun()
